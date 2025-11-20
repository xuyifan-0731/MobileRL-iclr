# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import logging
import os
import time
import warnings
from datetime import timedelta

import numpy as np
import ray
import torch
import torch.distributed
from codetiming import Timer
from megatron.core import parallel_state as mpu
from omegaconf import DictConfig
from torch.distributed import init_device_mesh

from verl import DataProto
from verl.models.mcore import get_mcore_weight_converter
from verl.protocol import list_of_dict_to_dict_of_list, DataProtoItem
from verl.single_controller.base.decorator import Dispatch, register, Execute
from verl.single_controller.base.megatron.worker import MegatronWorker
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.device import get_device_name
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
)
from verl.utils.model import load_mcore_dist_weights, load_megatron_gptmodel_weights
from verl.workers.abstract import AbstractCriticWorker, AbstractActorWorker, AbstractRolloutWorker
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.agentic.fsdp_sgl import FSDPSGLShardingManager
from verl.workers.critic.megatron_critic import MegatronPPOCritic
from verl.workers.reward_model.megatron.reward_model import MegatronRewardModel

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def set_random_seed(seed):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)
    # FIXME: torch cumsum not support deterministic (used in vllm sampler),
    # https://github.com/pytorch/pytorch/issues/89492
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class ActorRolloutRefWorker(MegatronWorker, AbstractActorWorker, AbstractRolloutWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config

        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
            torch.cuda.set_device(rank)

            if self.config.actor.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.actor.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.actor.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.actor.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.actor.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

            print(f"nodedup MGT TOPO {torch.distributed.get_rank()=} {mpu.get_all_ranks()=}")

        set_random_seed(seed=self.config.actor.megatron.seed)

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        # will support other offload later
        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False

        # normalize config
        if self._is_actor or self._is_rollout:
            self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
            if self.config.actor.get("ppo_micro_batch_size", None):
                self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size

            self._is_offload_param = self.config.actor.megatron.get("param_offload", False)
            self._is_offload_grad = self.config.actor.megatron.get("grad_offload", False)
            self._is_offload_optimizer = self.config.actor.megatron.get("optimizer_offload", False)
        elif self._is_ref:
            if self.config.ref.get("log_prob_micro_batch_size", None):
                self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
            self._ref_is_offload_param = self.config.ref.megatron.get("param_offload", False)

        self.param_on_gpu = True
        self.optimizer_on_gpu = True

    def _build_model_optimizer(self, model_path, optim_config, override_model_config, override_transformer_config):
        from megatron.core.models.gpt.gpt_model import ModelType

        from verl.utils.megatron.optimizer import get_megatron_optimizer
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config
        from verl.utils.model import get_generation_config, print_model_size

        self._init_hf_config_and_tf_config(model_path, self.dtype, override_model_config, override_transformer_config, self.config.model.get("trust_remote_code", False))
        self.generation_config = get_generation_config(self.local_path)

        def megatron_actor_model_provider(pre_process, post_process):
            from verl.models.mcore import init_mcore_model

            parallel_model = init_mcore_model(self.tf_config, self.hf_config, pre_process, post_process, pp_layers=self.config.actor.megatron.pp_layers if self._is_actor else self.config.ref.megatron.get("pp_layers"), share_embeddings_and_output_weights=self.share_embeddings_and_output_weights, value=False, freeze_moe_router=override_model_config.get("moe_config", {}).get("freeze_moe_router", False))
            parallel_model.cuda()
            return parallel_model

        assert self.role in ["actor", "ref"], f"wrong role {self.role}"

        # Step 3: initialize the megatron model
        if self._is_actor:
            actor_module = get_model(
                megatron_actor_model_provider,
                wrap_with_ddp=True,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            )
            print(f"actor_module: {len(actor_module)}")
            if self.config.actor.load_weight:
                if self.config.actor.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(actor_module, self.config.actor.megatron.dist_checkpointing_path, is_value_model=False)
                else:
                    load_megatron_gptmodel_weights(self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False)

            if self.rank == 0:
                print_model_size(actor_module[0])
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)
        elif self._is_ref:
            print(f"self.config.ref.load_weight: {self.config.ref.load_weight}")
            ref_module = get_model(
                model_provider_func=megatron_actor_model_provider,
                model_type=ModelType.encoder_or_decoder,
                wrap_with_ddp=False,
                use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
            )

            if self.config.ref.load_weight:  # should align with the actor:
                assert self.config.actor.load_weight == self.config.ref.load_weight
                print("load ref weight start")
                if self.config.ref.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(ref_module, self.config.ref.megatron.dist_checkpointing_path, is_value_model=False)
                else:
                    load_megatron_gptmodel_weights(self.config, self.hf_config, ref_module, params_dtype=self.dtype, is_value_model=False)
            log_gpu_memory_usage("After ref module init", logger=logger)
            return ref_module, self.hf_config

        # TODO: add more optimizer args into config
        if self._is_actor:
            optim_config = init_megatron_optim_config(optim_config)
            actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config)
        else:
            optim_config = None
            actor_optimizer = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        return actor_module, actor_optimizer, self.hf_config, optim_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.weight",
        }
        if self.config.rollout.name == "vllm":
            from torch.distributed.device_mesh import init_device_mesh

            from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
            from verl.workers.sharding_manager.megatron_vllm import MegatronVLLMShardingManager

            # we will reorganize their weight format when resharding from actor to rollout.

            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "tp"])
            log_gpu_memory_usage("Before building vllm rollout", logger=None)

            local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get('use_shm', False))
            if vllm_mode == "customized":
                rollout = vLLMRollout(
                    actor_module=self.actor_module,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                )
            elif vllm_mode == "spmd":
                rollout = vLLMRollout(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                )
            log_gpu_memory_usage("After building vllm rollout", logger=logger)

            # perform weight resharding between actor and rollout
            from verl.models.mcore import get_mcore_weight_converter

            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
            sharding_manager = MegatronVLLMShardingManager(
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                layer_name_mapping=layer_name_mapping,
                actor_module=self.actor.actor_module,
                weight_converter=weight_converter,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif self.config.rollout.name == "async":
            from verl.workers.agentic.async_rollout import AsyncRollout
            from verl.workers.agentic.fsdp_sgl import FSDPSGLShardingManager
            from verl.models.mcore import get_mcore_weight_converter
            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            rollout_device_mesh = init_device_mesh(get_device_name(), mesh_shape=(dp, infer_tp), mesh_dim_names=("dp", "tp"))
            cpu_group = torch.distributed.new_group(backend="gloo", timeout=timedelta(minutes=60))
            local_path = copy_to_local(self.config.model.path)
            rollout = AsyncRollout(
                model_path=local_path,
                config=self.config.rollout,
                device_mesh=rollout_device_mesh,
            )
            if "actor" in self.role:
                weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
                module = self.actor.actor_module
            else:
                weight_converter = None
                module = None
                self.actor_optimizer = None
                self.actor_lr_scheduler = None
                self.actor_model_config = None
                self.tf_config = None
            sharding_manager = FSDPSGLShardingManager(
                module=module,
                inference_engine=rollout.engine,
                rollout_worker=rollout,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                cpu_group=cpu_group,
                role=self.role,
                rollout_count=self.config.get("rollout_count"),
                exchange_size=self.config.get("exchange_size"),
                config=self.config,
                mode="mgt",
                transformer_config=self.tf_config,
                layer_name_mapping=layer_name_mapping,
                weight_converter=weight_converter,
            )

        elif self.config.rollout.name in ["sglang", "sglang_async"]:
            if self.config.rollout.name == "sglang_async":
                warnings.warn(
                    "'sglang_async' has been deprecated and merged into 'sglang'. Please use 'sglang' going forward.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            from verl.workers.rollout.sglang_rollout import SGLangRollout

            # However, due to verl's setting, the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.megatron_sglang import MegatronSGLangShardingManager

            infer_tp = self.config.rollout.tensor_model_parallel_size
            dp = self.world_size // infer_tp
            assert self.world_size % infer_tp == 0, f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
            rollout_device_mesh = init_device_mesh("cpu", mesh_shape=(dp, infer_tp, 1), mesh_dim_names=("dp", "tp", "pp"))

            local_path = copy_to_local(self.config.model.path)
            log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=None)
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
                device_mesh=rollout_device_mesh,
            )
            log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=None)

            from verl.models.mcore import get_mcore_weight_converter

            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
            sharding_manager = MegatronSGLangShardingManager(
                actor_module=self.actor.actor_module,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                layer_name_mapping=layer_name_mapping,
                weight_converter=weight_converter,
                device_mesh=rollout_device_mesh,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)
        else:
            raise NotImplementedError("Only vllmRollout is supported with Megatron now")

        return rollout, sharding_manager

    @GPUMemoryLogger(role="actor_model_to_gpu", logger=None)
    def model_to_gpu(self):
        if self._is_offload_param and not self.param_on_gpu:
            if self._is_actor:
                load_megatron_model_to_gpu(self.actor_module)
            elif self._is_ref:
                load_megatron_model_to_gpu(self.ref_module, load_grad=False)
            self.param_on_gpu = True
        else:
            print(f"skipping actor model to gpu because {self._is_offload_param=} and {self.param_on_gpu=}")

    @GPUMemoryLogger(role="actor_model_to_cpu", logger=None)
    def model_to_cpu(self):
        if self._is_offload_param and self.param_on_gpu:
            if self._is_actor:
                offload_megatron_model_to_cpu(self.actor_module)
            if self._is_ref:
                offload_megatron_model_to_cpu(self.ref_module)
            self.param_on_gpu = False
        else:
            print(f"skipping actor model to cpu because {self._is_offload_param=} and {self.param_on_gpu=}")

    @GPUMemoryLogger(role="actor_optimizer_to_gpu", logger=None)
    def optimizer_to_gpu(self):
        if self._is_actor and self._is_offload_optimizer and not self.optimizer_on_gpu:
            load_megatron_optimizer(self.actor_optimizer)
            self.optimizer_on_gpu = True
        else:
            print(f"skipping actor optimizer to gpu because {self._is_actor=} {self._is_offload_optimizer=} and {self.optimizer_on_gpu=}")


    @GPUMemoryLogger(role="actor_optimizer_to_cpu", logger=None)
    def optimizer_to_cpu(self):
        if self._is_actor and self._is_offload_optimizer and self.optimizer_on_gpu:
            offload_megatron_optimizer(self.actor_optimizer)
            self.optimizer_on_gpu = False
        else:
            print(f"skipping actor optimizer to cpu because {self._is_actor=} {self._is_offload_optimizer=} and {self.optimizer_on_gpu=}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @GPUMemoryLogger(role="actor_init_model", logger=None)
    def init_model(self):
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        from omegaconf import OmegaConf

        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        if self._is_actor:
            override_transformer_config = OmegaConf.to_container(self.config.actor.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True)
        elif self._is_ref:
            override_transformer_config = OmegaConf.to_container(self.config.ref.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True)
        else:
            override_transformer_config = {}
        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        if self._is_actor:
            # we need the model for actor only
            optim_config = self.config.actor.optim
            self.actor_module, self.actor_optimizer, self.actor_model_config, self.actor_optim_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=optim_config,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            log_gpu_memory_usage("After init actor model and optimizer", logger=logger)
            if self._is_offload_param:
                self.model_to_cpu()
            if self._is_offload_optimizer:
                self.optimizer_to_cpu()

        if self._is_actor:
            self.actor = MegatronPPOActor(
                config=self.config.actor,
                model_config=self.actor_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.actor_module,
                actor_optimizer=self.actor_optimizer,
            )
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)

        if self._is_rollout:
            self.rollout, self.sharding_manager = self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
            log_gpu_memory_usage("After rollout init", logger=logger)
        elif self._is_actor:
            layer_name_mapping = {
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.weight",
            }
            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
            tp = self.config.actor.megatron.tensor_model_parallel_size
            pp = self.config.actor.megatron.pipeline_model_parallel_size
            dp = self.world_size // tp // pp
            self.sharding_manager = FSDPSGLShardingManager(
                module=self.actor.actor_module,
                inference_engine=None,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                # doesn't matter, just need to know who's head
                device_mesh=init_device_mesh("cuda", mesh_shape=(pp, dp, tp), mesh_dim_names=["pp", "dp", "tp"]),
                role=self.role,
                rollout_count=self.config.get("rollout_count"),
                exchange_size=self.config.get("exchange_size"),
                config=self.config,
                mode="mgt",
                transformer_config=self.tf_config,
                layer_name_mapping=layer_name_mapping,
                weight_converter=weight_converter,
            )

        if self._is_ref:
            self.ref_module, self.ref_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=None,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            log_gpu_memory_usage("After ref model init", logger=logger)
            self.ref_policy = MegatronPPOActor(
                config=self.config.ref,
                model_config=self.ref_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.ref_module,
                actor_optimizer=None,
            )
            if self._ref_is_offload_param:
                self.model_to_cpu()

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_mananager = MegatronCheckpointManager(
                config=self.config,
                model_config=self.actor_model_config,
                role="actor",
                model=self.actor_module,
                arch=self.architectures[0],
                hf_config=self.hf_config,
                param_dtype=self.param_dtype,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                tokenizer=self.tokenizer,
                optimizer=self.actor_optimizer,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After actor_init_model finish", logger=logger)

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @GPUMemoryLogger(role="update_actor", logger=None)
    def update_actor(self, data: DataProto, offload=True):
        assert self._is_actor
        if self._is_offload_param:
            self.model_to_gpu()
        if self._is_offload_optimizer:
            self.optimizer_to_gpu()
        data.batch = data.batch.cuda()

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        dataloader = self.actor.make_minibatch_iterator(data=data)
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        metrics["perf/flops/actor"] = estimated_flops * self.config.actor.ppo_epochs / self.world_size
        metrics["perf/promised_flops/actor"] = promised_flops

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if offload:
            if self._is_offload_param:
                self.model_to_cpu()
            if self._is_offload_optimizer:
                self.optimizer_to_cpu()

            torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_async_rollout_workers(self, concurrency, queue_in, buffer_out):
        assert self._is_rollout
        self.rollout.start_async_workers(concurrency, queue_in, buffer_out)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @GPUMemoryLogger(role="sync_params", logger=None)
    def sync_params(self, no_rnd=False):
        if self._is_offload_param:
            self.model_to_gpu()

        self.sharding_manager.sync_params(no_rnd)

        # after parameters sync with rollout, offload actor model to CPU
        if self._is_offload_param:
            self.model_to_cpu()
        if self._is_offload_optimizer:
            self.optimizer_to_cpu()

    @register(dispatch_mode=Dispatch.RANK_ZERO, execute_mode=Execute.RANK_ZERO)
    def get_async_seqs(self, buffer_out, minimum, multiple):
        items = ray.get(buffer_out.get.remote(minimum=minimum, multiple=multiple))
        results, others = list(zip(*items))

        def collate_fn(x: list[DataProtoItem]):
            batch = []
            non_tensor_batch = []
            for data in x:
                batch.append(data.batch)
                non_tensor_batch.append(data.non_tensor_batch)
            batch = torch.stack(batch).contiguous()
            non_tensor_batch = list_of_dict_to_dict_of_list(non_tensor_batch)
            for key, val in non_tensor_batch.items():
                non_tensor_batch[key] = np.array(val, dtype=object)
            return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

        other = collate_fn(others)
        ret = self.rollout.make_batch(results)
        ret.union(other)
        return ret

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @GPUMemoryLogger(role="generate_sequences", logger=None)
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        # if self._is_offload_param:
        #     load_megatron_model_to_gpu(self.actor_module)
        #     log_gpu_memory_usage("After load actor params during generate_sequences", logger=logger)
        prompts.batch = prompts.batch.cuda()
        # meta_info = {
        #     "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
        #     "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        # }
        # prompts.meta_info.update(meta_info)

        # if self._is_offload_param:
        #     offload_megatron_model_to_cpu(self.actor_module)
        # if self._is_offload_optimizer:
        #     offload_megatron_optimizer(self.actor_optimizer)
        log_gpu_memory_usage("After entering sharding manager", logger=logger)

        prompts = self.sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(inputs=prompts)
        output = self.sharding_manager.postprocess_data(output)
        output.check_consistency()

        self.sharding_manager.offload()

        output = output.to("cpu")
        # clear kv cache
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @GPUMemoryLogger(role="compute_ref_log_prob", logger=None)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        if self._ref_is_offload_param:
            self.model_to_gpu()
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        data = data.to(torch.cuda.current_device())
        output, _ = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
        output = DataProto.from_dict(tensors={"ref_log_prob": output})
        output = output.to("cpu")
        if self._ref_is_offload_param:
            self.model_to_cpu()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @GPUMemoryLogger(role="compute_log_prob", logger=None)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            self.model_to_gpu()
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        data = data.to(torch.cuda.current_device())
        output, entropys = self.actor.compute_log_prob(data=data, calculate_entropy=True)
        output = DataProto.from_dict(tensors={"old_log_probs": output, "entropys": entropys}, meta_info={"temperature": self.config.rollout.temperature})
        output = output.to("cpu")
        # clear kv cache
        if self._is_offload_param:
            self.model_to_cpu()
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        if self._is_offload_param:
            self.model_to_gpu()
        self.checkpoint_mananager.load_checkpoint(local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)
        if self._is_offload_param:
            self.model_to_cpu()
        if self._is_offload_optimizer:
            self.optimizer_to_cpu()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_pretrained_model(self, checkpoint_path, del_local_after_load=True):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            self.model_to_gpu()
        self.checkpoint_mananager.save_checkpoint(local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)
        torch.distributed.barrier()
        if self._is_offload_param:
            self.model_to_cpu()


class CriticWorker(MegatronWorker, AbstractCriticWorker):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
            torch.cuda.set_device(rank)

            if self.config.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        # set FSDP offload params
        self._is_offload_param = self.config.megatron.param_offload
        self._is_offload_optimizer = self.config.megatron.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        if self.config.get("ppo_micro_batch_size", None):
            self.config.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size

        self.param_on_gpu = True
        self.optimizer_on_gpu = True


    def _build_critic_model_optimizer(self, model_path, optim_config, override_model_config, override_transformer_config):
        from megatron.core.models.gpt.gpt_model import ModelType

        from verl.utils.megatron.optimizer import get_megatron_optimizer
        from verl.utils.megatron_utils import get_model, init_megatron_optim_config
        from verl.utils.model import print_model_size

        self._init_hf_config_and_tf_config(model_path, self.dtype, override_model_config, override_transformer_config, self.config.model.get("trust_remote_code", False))

        def megatron_critic_model_provider(pre_process, post_process):
            from verl.models.mcore import init_mcore_model

            parallel_model = init_mcore_model(self.tf_config, self.hf_config, pre_process, post_process, pp_layers=self.config.megatron.pp_layers, share_embeddings_and_output_weights=False, value=True, freeze_moe_router=override_model_config.get("moe_config", {}).get("freeze_moe_router", False))
            parallel_model.cuda()
            return parallel_model

        # Step 3: initialize the megatron model
        critic_module = get_model(
            model_provider_func=megatron_critic_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # critic_module = nn.ModuleList(critic_module)

        if self.config.load_weight:
            t0 = time.time()
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(critic_module, self.config.megatron.dist_checkpointing_path, is_value_model=True)
            else:
                load_megatron_gptmodel_weights(self.config, self.hf_config, critic_module, params_dtype=self.dtype, is_value_model=True)
            t1 = time.time()
            if torch.distributed.get_rank() == 0:
                print(f"critic load_weight time: {t1 - t0}")
        if self.rank == 0:
            print_model_size(critic_module[0])

        # TODO: add more optimizer args into config
        optim_config = init_megatron_optim_config(optim_config)
        critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config)
        torch.cuda.empty_cache()
        return critic_module, critic_optimizer, self.hf_config, optim_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @GPUMemoryLogger(role="critic_init_model", logger=None)
    def init_model(self):
        # create critic
        from omegaconf import OmegaConf

        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_transformer_config = OmegaConf.to_container(self.config.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True)
        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        self.critic_module, self.critic_optimizer, self.critic_model_config, critic_optimizer_config = self._build_critic_model_optimizer(
            model_path=self.config.model.path,
            optim_config=self.config.optim,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        log_gpu_memory_usage("After init critic model and optimizer", logger=logger)
        if self._is_offload_param:
            self.model_to_cpu()
        if self._is_offload_optimizer:
            self.optimizer_to_cpu()

        self.critic = MegatronPPOCritic(
            config=self.config,
            model_config=self.critic_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            critic_optimizer_config=critic_optimizer_config,
        )
        log_gpu_memory_usage("After MegatronPPOCritic init", logger=logger)
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            model_config=self.critic_model_config,
            role="critic",
            model=self.critic_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=False,
            tokenizer=self.tokenizer,
            optimizer=self.critic_optimizer,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            checkpoint_contents=self.config.checkpoint.contents,
        )

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @GPUMemoryLogger(role="compute_values", logger=None)
    def compute_values(self, data: DataProto, offload=True):
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data = data.to(torch.cuda.current_device())
        if self._is_offload_param:
            self.model_to_gpu()
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={"values": values})
        output = output.to("cpu")
        if offload:
            self.model_to_cpu()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @GPUMemoryLogger(role="critic_model_to_gpu", logger=None)
    def model_to_gpu(self):
        if self._is_offload_param and not self.param_on_gpu:
            load_megatron_model_to_gpu(self.critic_module)
            self.param_on_gpu = True
        else:
            print(f"skipping critic model to gpu because {self._is_offload_param=} and {self.param_on_gpu=}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @GPUMemoryLogger(role="critic_model_to_cpu", logger=None)
    def model_to_cpu(self):
        if self._is_offload_param and self.param_on_gpu:
            offload_megatron_model_to_cpu(self.critic_module)
            self.param_on_gpu = False
        else:
            print(f"skipping critic model to cpu because {self._is_offload_param=} and {self.param_on_gpu=}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @GPUMemoryLogger(role="critic_optimizer_to_gpu", logger=None)
    def optimizer_to_gpu(self):
        if self._is_offload_optimizer and not self.optimizer_on_gpu:
            load_megatron_optimizer(self.critic_optimizer)
            self.optimizer_on_gpu = True
        else:
            print(f"skipping critic optimizer to gpu because {self._is_offload_optimizer=} and {self.optimizer_on_gpu=}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @GPUMemoryLogger(role="critic_optimizer_to_cpu", logger=None)
    def optimizer_to_cpu(self):
        if self._is_offload_optimizer and self.optimizer_on_gpu:
            offload_megatron_optimizer(self.critic_optimizer)
            self.optimizer_on_gpu = False
        else:
            print(f"skipping critic optimizer to cpu because {self._is_offload_optimizer=} and {self.optimizer_on_gpu=}")

    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    @GPUMemoryLogger(role="update_critic", logger=None)
    def update_critic(self, data: DataProto, offload=True):
        data = data.to(torch.cuda.current_device())

        if self._is_offload_param:
            self.model_to_gpu()
        if self._is_offload_optimizer:
            self.optimizer_to_gpu()

        dataloader = self.critic.make_minibatch_iterator(data)
        with Timer(name="update_critic", logger=None) as timer:
            metrics = self.critic.update_critic(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
        metrics["perf/flops/critic"] = estimated_flops * self.config.ppo_epochs / self.world_size
        metrics["perf/promised_flops/critic"] = promised_flops
        output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param and offload:
            self.model_to_cpu()
        if self._is_offload_optimizer and offload:
            self.optimizer_to_cpu()
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        if self._is_offload_param:
            self.model_to_gpu()
        self.checkpoint_mananager.load_checkpoint(local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load)
        if self._is_offload_param:
            self.model_to_cpu()
        if self._is_offload_optimizer:
            self.optimizer_to_cpu()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_steps=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            self.model_to_gpu()
        self.checkpoint_mananager.save_checkpoint(local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_steps, max_ckpt_to_keep=max_ckpt_to_keep)
        if self._is_offload_param:
            self.model_to_cpu()


class RewardModelWorker(MegatronWorker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel startegy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(rank)

            if self.config.megatron.sequence_parallel:
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        set_random_seed(seed=self.config.megatron.seed)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_rm_model(self, model_path, override_model_config, override_transformer_config):
        from megatron.core.models.gpt.gpt_model import ModelType

        from verl.utils.megatron_utils import get_model

        self._init_hf_config_and_tf_config(model_path, self.dtype, override_model_config, override_transformer_config, self.config.model.get("trust_remote_code", False))

        def megatron_rm_model_provider(pre_process, post_process):
            from verl.models.mcore import init_mcore_model

            parallel_model = init_mcore_model(
                self.tf_config,
                self.hf_config,
                pre_process,
                post_process,
                share_embeddings_and_output_weights=False,
                value=True,
            )
            parallel_model.cuda()
            return parallel_model

        # Step 3: initialize the megatron model
        reward_model = get_model(
            model_provider_func=megatron_rm_model_provider,
            model_type=ModelType.encoder_or_decoder,
            wrap_with_ddp=False,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # reward_model = nn.ModuleList(reward_model)

        if self.config.load_weight:
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(reward_model, self.config.megatron.dist_checkpointing_path, is_value_model=True)
            else:
                load_megatron_gptmodel_weights(self.config, self.hf_config, reward_model, params_dtype=self.dtype, is_value_model=True)

        # TODO: add more optimizer args into config
        torch.cuda.empty_cache()
        return reward_model, self.hf_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # create critic
        from omegaconf import OmegaConf

        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_transformer_config = OmegaConf.to_container(self.config.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True)

        use_shm = self.config.model.get('use_shm', False)
        sft_tokenizer_local_path = copy_to_local(self.config.model.input_tokenizer, use_shm=use_shm)
        sft_tokenizer = hf_tokenizer(sft_tokenizer_local_path)
        rm_tokenizer_path = self.config.model.get("rm_tokenizer", None)
        rm_tokenizer = None
        if rm_tokenizer_path is not None:
            rm_tokenizer_local_path = copy_to_local(rm_tokenizer_path, use_shm=use_shm)
            rm_tokenizer = hf_tokenizer(rm_tokenizer_local_path)

        self.param_dtype = torch.bfloat16
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        reward_model_module, reward_model_config = self._build_rm_model(
            model_path=self.config.model.path,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        # FIXME(sgm): reward model param offload is implemented in MegatronRewardModel
        # should be implemented in workers
        self.rm = MegatronRewardModel(
            config=self.config,
            reward_model_module=reward_model_module,
            model_config=reward_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            sft_tokenizer=sft_tokenizer,
            rm_tokenizer=rm_tokenizer,
        )

    # TODO: reward model use itself tokenizer instead of sft tokenizer
    # the input_ids, responses, attention_mask and position_ids may be different!
    @register(dispatch_mode=Dispatch.MEGATRON_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        data.meta_info["micro_batch_size"] = self.config.micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data = data.to(torch.cuda.current_device())
        output = self.rm.compute_reward(data)
        output = output.to("cpu")
        return output
