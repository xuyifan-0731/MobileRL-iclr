import asyncio
import functools
import gc
import logging
import os
import random
import time
from typing import List, Any, Optional

import numpy as np
import torch
# from sglang.srt.entrypoints.engine import Engine
from tensordict import TensorDict
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import _group_or_default_group, _canonicalize_group_rank, _warn_not_in_group, \
    _rank_not_in_group, _get_object_coll_device, _object_to_tensor, broadcast, _tensor_to_object
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
from tqdm import tqdm

from verl import DataProto
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors, all_gather_dict_non_tensors,
                                         broadcast_dict_non_tensor)
from ..sharding_manager.base import BaseShardingManager
from ...utils.fsdp_utils import fsdp_version

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))


def broadcast_object_list(
        object_list: List[Any],
        src: Optional[int] = None,
        group: Optional[ProcessGroup] = None,
        device: Optional[torch.device] = None,
        group_src: Optional[int] = None,
):
    """
    Broadcasts picklable objects in ``object_list`` to the whole group.

    Similar to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
            Source rank is based on global process group (regardless of ``group`` argument)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before broadcasting. Default is ``None``.
        group_src (int): Source rank on ``group``.  Must not specify one of ``group_src``
            and ``src`` but not both.

    Returns:
        ``None``. If rank is part of the group, ``object_list`` will contain the
        broadcasted objects from ``src`` rank.

    .. note:: For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. note:: Note that this API differs slightly from the :func:`broadcast`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. warning::
        :func:`broadcast_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`broadcast_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`broadcast` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     objects = [None, None, None]
        >>> # Assumes backend is not NCCL
        >>> device = torch.device("cpu")
        >>> dist.broadcast_object_list(objects, src=0, device=device)
        >>> objects
        ['foo', 12, {1: 2}]
    """
    group = _group_or_default_group(group)
    if src is None and group_src is None:
        src = 0
    group_src = _canonicalize_group_rank(group, src, group_src, return_global=False)
    if _rank_not_in_group(group):
        _warn_not_in_group("broadcast_object_list")
        return

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is default to ``None``
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
    # case it is not ``None`` we move the size and object tensors to be
    # broadcasted to this device.
    current_device = device or _get_object_coll_device(group)
    my_group_rank = group.rank()
    # Serialize object_list elements to tensors on src rank.
    if my_group_rank == group_src:
        tensor_list, size_list = zip(
            *[_object_to_tensor(obj, current_device, group) for obj in object_list]
        )
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(
            len(object_list), dtype=torch.long, device=current_device
        )

    # Broadcast object sizes
    broadcast(object_sizes_tensor, group_src=group_src, group=group)

    # Concatenate and broadcast serialized object tensors
    # Note: torch.cat will do an extra memory copy to the current device, if the tensor_list
    # has only one element, we can skip the copy.
    if my_group_rank == group_src:
        if len(tensor_list) == 1:  # type: ignore[possibly-undefined]
            object_tensor = tensor_list[0]
        else:
            object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(  # type: ignore[call-overload]
            torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
            dtype=torch.uint8,
            device=current_device,
        )

    broadcast(object_tensor, group_src=group_src, group=group)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_group_rank != group_src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset: offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size, group)


def gc_and_clear_cuda(func):
    name = func.__name__

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            log_gpu_memory_usage(f"before cleaning {name}")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            log_gpu_memory_usage(f"after cleaning {name}")

    return wrapper


class FSDPSGLShardingManager(BaseShardingManager):
    def __init__(
        self,
        module,
        inference_engine,
        model_config,
        config,
        rollout_worker=None,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        cpu_group=None,
        role: str = "actor_rollout",
        rollout_count: int = 0,
        exchange_size: int | float | None = None,
        mode: str = "fsdp",
        # mgt params
        transformer_config=None,
        layer_name_mapping=None,
        weight_converter=None,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.rollout_worker = rollout_worker
        self.model_config = model_config
        self.config = config
        self.device_mesh = device_mesh
        self.cpu_group = cpu_group
        self.exchange_size = exchange_size

        # Full params
        self.full_params = full_params
        self.role = role
        self.mode = mode

        # mgt params
        self.transformer_config = transformer_config
        self.layer_name_mapping = layer_name_mapping
        self.weight_converter = weight_converter

        if "actor" in role and mode == "fsdp" and fsdp_version(self.module) == 1:
            if full_params:
                FSDP.set_state_dict_type(
                    self.module,
                    state_dict_type=StateDictType.FULL_STATE_DICT,
                    state_dict_config=FullStateDictConfig()
                )
            else:
                FSDP.set_state_dict_type(
                    self.module,
                    state_dict_type=StateDictType.SHARDED_STATE_DICT,
                    state_dict_config=ShardedStateDictConfig()
                )

        addr = os.environ["MLP_WORKER_0_HOST"]
        port = 9375
        from sglang.srt.utils import init_custom_process_group
        if role == "actor":
            if device_mesh.get_rank() == 0:
                self.update_weight_pg: ProcessGroup = init_custom_process_group(
                    backend="nccl",
                    init_method=f"tcp://{addr}:{port}",
                    world_size=rollout_count + 1,
                    rank=0,
                    group_name="update_weight_group",
                )
        if role == "rollout":
            assert rollout_count == device_mesh.size(), f"{rollout_count=}, {device_mesh.size()=}"
            self.update_weight_pg = init_custom_process_group(
                backend="nccl",
                init_method=f"tcp://{addr}:{port}",
                world_size=rollout_count + 1,
                rank=1 + device_mesh.get_rank(),
                group_name="update_weight_group",
            )

        # mark stale
        self.stale = False
        if stale_ratio := self.config.get("stale_ratio"):
            self.stale = device_mesh.get_local_rank("dp") < device_mesh["dp"].size() * stale_ratio

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None and "rollout" in role:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        torch.distributed.barrier()

    @gc_and_clear_cuda
    def sync_params(self, no_rnd=False):
        from sglang.srt.model_executor.model_runner import LocalSerializedTensor
        from sglang.srt.patch_torch import monkey_patch_torch_reductions
        from sglang.srt.utils import MultiprocessingSerializer
        tp_rank = self.device_mesh.get_local_rank("tp") if "tp" in self.device_mesh.mesh_dim_names else 0
        dp_rank = self.device_mesh.get_local_rank("dp")
        if "actor" in self.role and self.mode == "fsdp":
            st = self.module.state_dict()
            k, v = next(iter(st.items()))
            device = v.device
            target_device = torch.device("cpu") if self.config.get("cpu_exchange") else device
            tensor_list = []
            for k, v in tqdm(st.items(), desc="tensor list"):
                if isinstance(v, DTensor):
                    v = v.full_tensor()
                if dp_rank == 0:
                    v_bf16 = v.to(dtype=torch.bfloat16)
                    v_target = v_bf16.to(device=target_device)
                    tensor_list.append((k, v_target))
                    del v_bf16
            del st, v
            torch.cuda.empty_cache()
        if "actor" in self.role and self.mode == "mgt":
            from ...utils.megatron_utils import per_tensor_generator
            tensor_list = per_tensor_generator(
                self.module,
                self.model_config,
                self.weight_converter,
                self.transformer_config,
                self.layer_name_mapping,
            )
        if "rollout" in self.role and tp_rank == 0 and not self.role == "rollout":
            asyncio.set_event_loop(self.rollout_worker.event_loop)
            self.inference_engine.resume_memory_occupation()
        torch.cuda.synchronize()

        rnd = self.config.get("random_update", 1)
        rnd_same_params = self.config.get("random_same_params", True)
        rnd_full_params = self.config.get("random_full_params", False)
        if no_rnd:
            rnd = 1
        skip = rnd_full_params and rnd < 1 and random.random() > rnd or self.stale and not no_rnd

        def tensor_loader():
            for k, v in tensor_list:
                if rnd_same_params and not rnd_full_params and rnd < 1 and random.random() > rnd:
                    continue
                yield (k, v.to(device) if self.mode == "fsdp" else v), v.numel() * v.element_size()

        loader = tensor_loader()
        done = False
        loop_count = 0

        log_gpu_memory_usage(f'nodedup {self.role} Before sync model weights in sharding manager', logger=logger)

        if self.role == "rollout":
            self.rollout_worker.acquire_writer_lock()

        while not done:
            if self.role == "actor" and dp_rank != 0 or self.role == "actor_rollout" and tp_rank != 0:
                break
            each_loop_start_time = time.time()
            if self.role == "actor" and dp_rank == 0 or self.role == "actor_rollout" and tp_rank == 0:
                count = 0
                if self.exchange_size is None:
                    gpu_tensor_list = tensor_list
                    done = True
                else:
                    gpu_tensor_list = []
                    for item, size in loader:
                        gpu_tensor_list.append(item)
                        count += size
                        if count > self.exchange_size:
                            break
                    else:
                        done = True

            if self.role == "actor_rollout":
                if tp_rank == 0:
                    self.inference_engine.update_weights_from_tensor(gpu_tensor_list)
                    del gpu_tensor_list
            else:
                if self.role == "actor":
                    if self.device_mesh.get_rank() == 0:
                        # send tensor list
                        assert dp_rank == 0
                        descriptions = {k: (v.shape, v.dtype) for k, v in gpu_tensor_list}
                        lst = [descriptions]
                        torch.distributed.barrier(group=self.update_weight_pg)
                        broadcast_object_list(lst, group_src=0, group=self.update_weight_pg)
                        for _, v in gpu_tensor_list:
                            torch.distributed.broadcast(v, group_src=0, group=self.update_weight_pg)

                        # broadcast done
                        lst = [done]
                        broadcast_object_list(lst, group_src=0, group=self.update_weight_pg)
                    if dp_rank == 0:
                        del gpu_tensor_list
                else:
                    assert self.role == "rollout", f"unknown role {self.role=}"
                    # get tensor list
                    lst = [None]
                    rollout_tensor_list = []
                    torch.distributed.barrier(group=self.update_weight_pg)
                    broadcast_object_list(lst, group_src=0, group=self.update_weight_pg)
                    monkey_patch_torch_reductions()
                    for k, (shape, dtype) in lst[0].items():
                        v = torch.empty(shape, dtype=dtype, device='cuda')
                        torch.distributed.broadcast(v, group_src=0, group=self.update_weight_pg)
                        if not skip:
                            rollout_tensor_list.append((k, MultiprocessingSerializer.serialize(v)))
                    if tp_rank == 0:
                        gathered_list = [None] * self.device_mesh.size(1)
                    else:
                        gathered_list = None
                    torch.distributed.gather_object(
                        obj=rollout_tensor_list,
                        object_gather_list=gathered_list,
                        group_dst=0,
                        group=self.device_mesh.get_group("tp"),
                    )
                    gathered_list: list[list[tuple[str, bytes]]] | None

                    # receive broadcasted done
                    lst = [None]
                    broadcast_object_list(lst, group_src=0, group=self.update_weight_pg)
                    assert lst[0] is not None
                    done = lst[0]

                    if tp_rank == 0 and not skip:
                        serialized_tensor_list = [
                            (group[0][0], LocalSerializedTensor(values=[g[1] for g in group]))
                            for group in zip(*gathered_list)
                        ]
                        if not rnd_same_params and not rnd_full_params and rnd < 1:
                            serialized_tensor_list = [item for item in serialized_tensor_list if random.random() > rnd]

                        # do update
                        if self.rollout_worker.full_async:
                            task = self.inference_engine.async_update_weights_from_tensor(
                                serialized_tensor_list,
                                flush_cache=True,
                            )
                            asyncio.run_coroutine_threadsafe(task, self.rollout_worker.event_loop).result()
                        else:
                            self.inference_engine.update_weights_from_tensor(serialized_tensor_list)
                        del serialized_tensor_list

                    # clean up
                    del rollout_tensor_list, gathered_list
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            loop_consumed_time = time.time() - each_loop_start_time
            log_gpu_memory_usage(
                f'nodedup {self.role=} {dp_rank=} {tp_rank=} After loop {loop_count} {done=} {loop_consumed_time=} in sharding manager',
                logger=logger)
            loop_count += 1

        if "actor" in self.role:
            del tensor_list, loader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log_gpu_memory_usage(
            f'{self.role=} rank{self.device_mesh.get_rank()} After sync model weights in sharding manager',
            logger=logger,
        )

        if self.role == "rollout":
            self.rollout_worker.release_writer_lock()
        torch.distributed.barrier()

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None and "rollout" in self.role:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def offload(self):
        log_gpu_memory_usage('Before sglang offload in sharding manager', logger=logger)
        if "rollout" in self.role and self.device_mesh.get_local_rank("tp") == 0 and not self.rollout_worker.full_async:
            self.inference_engine.release_memory_occupation()
        log_gpu_memory_usage('After sglang offload in sharding manager', logger=logger)

        if "actor" in self.role:
            self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None and "rollout" in self.role:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        data.batch = allgather_dict_tensors(
            data.batch.contiguous(),
            size=self.device_mesh.size(1),
            group=self.device_mesh.get_group(1),
            dim=0,
        )
        data.non_tensor_batch = all_gather_dict_non_tensors(
            data.non_tensor_batch,
            size=self.device_mesh.size(1),
            group=self.device_mesh.get_group(1),
        )

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # prevent nccl timeout
        assert "rollout" in self.role, f"postprocess_data should only be called in rollout worker, but got {self.role=}"
        torch.distributed.barrier(self.cpu_group)
        tp_size = self.device_mesh["tp"].size()
        tp_rank = self.device_mesh.get_local_rank("tp")
        src_rank = self.device_mesh.get_local_rank("dp") * tp_size
        # obs metrics are dynamically acquired, so we should build a same shape tensor dynamically, communicate shapes and dtypes first
        if tp_rank == 0:
            description: dict = {k: (v.shape, v.dtype) for k, v in data.batch.items()}
            description['batch_size'] = data.batch.batch_size
            description['non_tensor_batch_keys'] = list(data.non_tensor_batch.keys())
            lst = [description]
        else:
            lst = [None]
        torch.distributed.broadcast_object_list(lst, src=src_rank, group=self.device_mesh.get_group("tp"))
        description = lst[0]
        if tp_rank != 0:
            batch_size = description.pop('batch_size')
            non_tensor_batch_keys = description.pop('non_tensor_batch_keys')
            batch = TensorDict(
                {k: torch.empty(shape, dtype=dtype, device='cuda') for k, (shape, dtype) in description.items()},
                batch_size=batch_size)
            non_tensor_batch = {k: [None] * int(batch_size[0]) for k in non_tensor_batch_keys}
        else:
            batch = data.batch
            non_tensor_batch = data.non_tensor_batch
        broadcast_dict_tensor(
            batch,
            src=src_rank,
            group=self.device_mesh.get_group("tp"),
        )
        broadcast_dict_non_tensor(
            non_tensor_batch,
            src=src_rank,
            group=self.device_mesh.get_group("tp"),
        )
        if tp_rank != 0:
            data = DataProto(batch=batch, non_tensor_batch={k: np.array(v) for k, v in non_tensor_batch.items()})
        if tp_size > 1:
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[tp_rank]
        return data
