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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os

import hydra
import ray
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from math import ceil

@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    OmegaConf.set_struct(config, False)
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", **dict(os.environ)}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        OmegaConf.register_new_resolver("eval", eval)
        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        n = config.actor_rollout_ref.rollout.n
        config.actor_rollout_ref.actor.ppo_mini_batch_size *= n
        config.critic.ppo_mini_batch_size *= n

        if config.trainer.get("hybrid_engine", True):
            role_worker_mapping = {
                Role.ActorRollout: ray.remote(actor_rollout_cls),
                Role.Critic: ray.remote(CriticWorker),
            }

            global_pool_id = "global_pool"
            print(f'config.trainer.nnodes: {config.trainer.nnodes}')
            print(f'config.trainer.n_gpus_per_node: {config.trainer.n_gpus_per_node}')
            resource_pool_spec = {
                global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }
            mapping = {
                Role.ActorRollout: global_pool_id,
                Role.Critic: global_pool_id,
                Role.RefPolicy: global_pool_id,
            }
        else:
            role_worker_mapping = {
                Role.Actor: ray.remote(ActorRolloutRefWorker),
                Role.Critic: ray.remote(CriticWorker),
                Role.Rollout: ray.remote(ActorRolloutRefWorker),
            }

            placement = config.trainer.placement
            split = placement.get("split", "horizontal")

            mapping = {}
            resource_pool_spec = {}
            gpus_per_node = config.trainer.n_gpus_per_node
            if split == "horizontal":
                for name, gpus in placement["pools"].items():
                    resource_pool_spec[name] = [gpus] * config.trainer.nnodes
            elif split == "vertical":
                counts = placement["pools"].values()
                total = sum(counts)
                allocation = [int(config.trainer.nnodes * count / total) for count in counts]
                print(f"{allocation=}")
                for name, count in zip(placement["pools"].keys(), allocation):
                    resource_pool_spec[name] = [gpus_per_node] * count
            else:
                raise ValueError(f"Unknown split type: {split}")
            print(f"{resource_pool_spec=}")
            for role, pool in placement["mapping"].items():
                mapping[Role[role]] = pool
            rollout_pool = mapping[Role.Rollout]
            rollout_count = sum(resource_pool_spec[rollout_pool])
            config.actor_rollout_ref.rollout_count = rollout_count

            print(f"{config.data.train_batch_size=}")
            print(f"{config.data.val_batch_size=}")

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset, AgenticDatasetOrigin

    if data_config.task_type == "gen_chat":
        spec = data_config.gen_chat
        train = spec.train
        val = spec.val
        train_dataset = AgenticDatasetOrigin(
            name=train.name,
            index_start=train.index_start,
            index_end=train.index_end,
        )
        val_dataset = AgenticDatasetOrigin(
            name=val.name,
            index_start=val.index_start,
            index_end=val.index_end,
        )
    elif data_config.task_type == "gen_chat_multi":
        ds_list = []
        ratio_list = []
        max_len = 0
        for train in data_config.gen_chat_multi.train:
            ds = AgenticDatasetOrigin(
                name=train.name,
                index_start=train.index_start,
                index_end=train.index_end,
            )
            ratio = getattr(train, "ratio", 1)
            ratio_list.append(ratio)
            if len(ds) > max_len:
                max_len = len(ds)
            ds_list.append(ds)
            print("name:{}, len:{}, max_len:{}, ratio:{}".format(ds.name, len(ds), max_len, ratio))
        if data_config.get("equal_concat"):
            new_ds_list = []
            for ds in ds_list:
                copy = round(max_len / len(ds)) - 1
                print(f"copying {ds.name} for {copy} times")
                for i in range(copy):
                    new_ds_list.append(ds)
            ds_list.extend(new_ds_list)
        elif any(r != 1 for r in ratio_list):
            new_ds_list = []
            unit = max(len(ds) / r for ds, r in zip(ds_list, ratio_list))
            for ds, r in zip(ds_list, ratio_list):
                target_samples = int(round(unit * r))
                copy_times = ceil(target_samples / len(ds)) - 1
                for _ in range(copy_times):
                    new_ds_list.append(ds)  
                print("dataset {} copy {} times".format(ds.name, copy_times))
            ds_list.extend(new_ds_list)
        train_dataset = ConcatDataset(ds_list)
        val_dataset = ConcatDataset([
            AgenticDatasetOrigin(
                name=val.name,
                index_start=val.index_start,
                index_end=val.index_end,
            ) for val in data_config.gen_chat_multi.val]
        )

    if "gen_chat" in data_config.task_type:
        if "train" in data_paths:
            return train_dataset
        else:
            return val_dataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
