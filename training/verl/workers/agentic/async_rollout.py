import os
import threading
import time
from functools import partial

import ray
import sglang as sgl
import torch.distributed
import uvicorn
from aiohttp import ClientSession
from fastapi import FastAPI
from omegaconf import DictConfig
from pydantic import BaseModel
from ray.util.queue import Queue
from sglang.srt.aio_rwlock import RWLock
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from tensordict import TensorDict
from torch.distributed import DeviceMesh
from tqdm import tqdm
from transformers import AutoProcessor

from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from .aio_rwlock import WriteEnforceRWLock
from .loops import retry_openai_chat_agent_loop
from .tasks import *
from ...protocol import collate_fn

import base64
from io import BytesIO
from PIL import Image
import random
import traceback
import numpy as np
import torch
import asyncio

def process_image(image, max_pixels, min_pixels):
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    elif isinstance(image, str):
        if image.startswith("data:image") or "," in image:
            if "," in image:
                image = image.split(",", 1)[1]
            image = Image.open(BytesIO(base64.b64decode(image)))
        else:
            image = Image.open(image)
    
    from verl.models.transformers.qwen2_vl import smart_resize
    height, width = smart_resize(image.height, image.width, min_pixels=min_pixels, max_pixels=max_pixels)
    image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

class AsyncRollout(BaseRollout):
    def __init__(self, model_path, tokenizer, processor, config: DictConfig, device_mesh: DeviceMesh):
        super().__init__()
        torch.distributed.barrier()
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        self.tp_rank = device_mesh.get_local_rank(1)
        self.tp_size = device_mesh.size(1)
        cuda_visible_device = os.environ["CUDA_VISIBLE_DEVICES"]
        visible_devices: list[str | None] = [None] * device_mesh.size(1)
        torch.distributed.all_gather_object(visible_devices, cuda_visible_device, group=device_mesh.get_group(1))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
        self.total_len = config.prompt_length + config.response_length
        torch.distributed.barrier()
        rng = random.Random(time.time() + torch.distributed.get_rank())
        self.tokenizer = tokenizer
        self.processor = processor
        if self.tp_rank == 0:
            self.engine = sgl.Engine(
                model_path=model_path,
                port=40000 + rng.randint(0, 1000),
                dtype=config.dtype,
                max_prefill_tokens=self.total_len,
                enable_memory_saver=config.enable_memory_saver,
                mem_fraction_static=config.gpu_memory_utilization,
                tp_size=device_mesh.size(1),
                enable_metrics=True,
                enable_cache_report=True,
                log_level="INFO",
                # chat_template=config.get("chat_template"),
                **({"max_total_tokens": config.engine_max_tokens} if config.get("engine_max_tokens") else {}),
            )
            self.engine.release_memory_occupation()
        else:
            self.engine = None
        self.engine: sgl.srt.entrypoints.engine.Engine | None
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_device
        torch.distributed.barrier()
        self.config = config
        self.task_type = config.task_type
        self.sampling_params = dict(config.sampling_params)
        self.sampling_params.update({
            "skip_special_tokens": False,
        })
        if self.config.get("val_sampling_params"):
            self.val_sampling_params = dict(self.config.val_sampling_params)
            self.val_sampling_params.update({
                "skip_special_tokens": False,
            })
        else:
            self.val_sampling_params = None
        self.rw_lock = WriteEnforceRWLock() if self.config.get("use_force_cancel", False) else RWLock()
        self.event_loop = asyncio.get_event_loop()
        # for full async
        self.queue_out = Queue()
        self.full_async = False
        self.tasks = []
        self.loop_thread = None

        if self.tp_rank == 0:
            self.app = FastAPI()
            self.register()
            host = ray._private.services.get_node_ip_address()
            port = 6000 + device_mesh.get_local_rank(0)
            url = f"http://{host}:{port}/gen_chat"
            # broadcast urls
            self.urls = [None] * device_mesh.size(0)
            torch.distributed.all_gather_object(self.urls, url, device_mesh.get_group(0))
            def start_server():
                uvicorn.run(self.app, host="0.0.0.0", port=port, log_level="debug")
            self._server_thread = threading.Thread(
                target=start_server,
                daemon=True,
            )
            self._server_thread.start()
            self.session = ClientSession(loop=self.event_loop)
        torch.distributed.barrier()
        time.sleep(10)

    async def gen_chat(self, ids, image_data=None, val: bool=False):
        image_data = [process_image(image, max_pixels=self.config.max_pixels, min_pixels=self.config.min_pixels) for image in image_data]
        retries_time = 0
        if val and self.val_sampling_params:
            sampling_params = self.val_sampling_params
        else:
            sampling_params = self.sampling_params
        while True:
            try:
                if isinstance(ids, str):
                    async with self.rw_lock.reader_lock:
                        ret = await self.engine.async_generate(
                            prompt=ids,
                            sampling_params=sampling_params,
                            return_logprob=True,
                            image_data=image_data,
                        )
                else:
                    async with self.rw_lock.reader_lock:
                        ret = await self.engine.async_generate(
                            input_ids=ids,
                            sampling_params=sampling_params,
                            return_logprob=True,
                            image_data=image_data,
                        )
                
                log_probs = ret["meta_info"]["output_token_logprobs"]
                log_ps = torch.tensor([lp for lp, _, _ in log_probs])
                ppl = torch.exp(-log_ps.mean()).item()
                if ppl < 2:
                    break
                else:
                    retries_time += 1
                    if retries_time > 3:
                        break
            except asyncio.CancelledError:
                print("gen chat cancelled")

        text = ret["text"]
        return text, log_probs

    def register(self):
        class Input(BaseModel):
            ids: list[int]
            image_data: list[str]
            val: bool = False

        @self.app.post("/gen_chat")
        def gen_chat(inp: Input):
            text, logprobs = asyncio.run_coroutine_threadsafe(self.gen_chat(inp.ids, inp.image_data, inp.val), self.event_loop).result()
            return {
                "text": text,
                "logprobs": logprobs,
            }

    async def crossed_gen_chat(self, ids, image_data=None, val: bool=False):
        url = random.choice(self.urls)
        payload = {"ids": ids, "val": val}
        if image_data:                                   # 有图才处理
            encoded_imgs = []
            for img in image_data:
                buf = BytesIO()
                img.save(buf, format="PNG")              # 也可换成 JPEG 等
                encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
                encoded_imgs.append("data:image/png;base64," + encoded)

            payload["image_data"] = encoded_imgs 
        try:
            async with ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json(content_type=None)
                    text = data["text"]
                    log_probs = data["logprobs"]
                    ret = text, log_probs
        except:
            traceback.print_exc()
            raise
        return ret

    async def async_acquire_writer_lock(self):
        if not self.full_async or self.tp_rank != 0:
            return
        await self.rw_lock.acquire_writer()

    async def async_release_writer_lock(self):
        if not self.full_async or self.tp_rank != 0:
            return
        await self.rw_lock.release_writer()

    def acquire_writer_lock(self):
        if not self.full_async or self.tp_rank != 0 or self.config.get("forget_lock"):
            return
        task = self.async_acquire_writer_lock()
        asyncio.run_coroutine_threadsafe(task, self.event_loop).result()
        if self.config.get("use_force_cancel"):
            # make sure its cancelled
            time.sleep(2)

    def release_writer_lock(self):
        if not self.full_async or self.tp_rank != 0 or self.config.get("forget_lock"):
            return
        task = self.async_release_writer_lock()
        asyncio.run_coroutine_threadsafe(task, self.event_loop).result()

    def generate_sequences(self, inputs: DataProto) -> DataProto | None:
        if self.tp_rank != 0:
            return None

        # starting rollout
        tasks = [self.make_task(item.to_dict(), val=True) for item in inputs]
        if self.full_async:
            # will hit this branch when in full async mode but validating
            sem = asyncio.Semaphore(len(self.tasks))
            async def wrapper(task):
                async with sem:
                    return await task

            async def coro():
                return await asyncio.gather(*[wrapper(t) for t in tasks])
            results = asyncio.run_coroutine_threadsafe(coro(), self.event_loop).result()
        else:
            results = self.event_loop.run_until_complete(asyncio.gather(*tasks))

        ret = self.make_batch(results, val=True)
        return ret

    def start_async_workers(self, concurrency, queue_in: Queue, buffer_out):
        self.full_async = True
        if self.tp_rank != 0:
            return
        for _ in range(concurrency):
            task = self.event_loop.create_task(self.infinite_task_worker(queue_in, buffer_out))
            self.tasks.append(task)

        import threading
        def run_loop():
            self.event_loop.run_forever()

        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()

    async def infinite_task_worker(self, queue_in: Queue, buffer_out):
        while True:
            try:
                item, other = await queue_in.get_async()
                result = await self.make_task(item.to_dict())
                if result is None:
                    continue
                await buffer_out.add.remote(result, other)
            except:
                traceback.print_exc()

    def make_task(self, item, val: bool=False):
        # choose function set
        url = self.config["base_url"]
        loop_fn, start_fn, gen_fn, obs_fn, end_fn = {
            "gen_chat": (
                partial(
                    retry_openai_chat_agent_loop,
                    incomplete_punishment=self.config.get("incomplete_punishment", 0),
                    tool_call_parser=self.config.get("tool_call_parser", "qwen25"),
                    max_pixels=self.config.get("max_pixels", 500000),
                ),
                partial(openai_chat_start, url=url),
                self.gen_chat if not self.config.get("crossed_gen", False) else self.crossed_gen_chat,
                partial(openai_chat_obs, url=url),
                partial(openai_chat_end, url=url)
            ),
        }[self.task_type]

        return self.event_loop.create_task(loop_fn(
            start_args=item,
            start_fn=start_fn,
            gen_fn=gen_fn,
            obs_fn=obs_fn,
            end_fn=end_fn,
            max_turns=self.config.max_turns,
            # max_length=self.total_len - 50,
            prompt_length=self.config.prompt_length - 50,
            response_length=self.config.response_length,
            tokenizer=self.tokenizer,
            processor=self.processor,
            rollout_trace_path=self.config.rollout_trace_path,
            image_only=self.config.image_only,
            val=val,
        ))

    def make_batch(self, results, val: bool=False):
        if val:
            batch_size = len(results)
            results = [sublist[-1] for sublist in results if sublist is not None]
        else:
            results = [item for sublist in results for item in sublist if item is not None] 
            batch_size = len(results)
        if batch_size == 0:
            return DataProto(batch=None, non_tensor_batch={}, meta_info={})
        device = torch.cuda.current_device()
        # make batch
        pad = self.tokenizer.pad_token_id
        max_len, prompt_len, response_len = self.total_len, self.config.prompt_length, self.config.response_length
        prompts_ids = torch.full((batch_size, prompt_len), pad, dtype=torch.long, device=device)
        responses = torch.full((batch_size, response_len), pad, dtype=torch.long, device=device)
        log_probs = torch.zeros((batch_size, response_len), dtype=torch.float, device=device)
        # loss_mask = torch.zeros((batch_size, max_len), dtype=torch.int, device=device)
        if "reward" in results[0]:
            rewards = torch.zeros((batch_size,), dtype=torch.float, device=device)
        else:
            rewards = None
        obs_metric_keys = set()
        for r in results:
            obs_metric_keys.update(r["obs_metrics"].keys())
        obs_metrics = {k: torch.full((batch_size, response_len), torch.nan, dtype=torch.float32, device=device) for k in obs_metric_keys}
        if "dapo_metrics" in results[0]:
            dapo_metrics = results[0]["dapo_metrics"]
        else:
            dapo_metrics = None

        all_multi_modal_inputs = []
        per_sample_resp_loss_masks = []
        
        has_image_data = False
        for i, r in enumerate(results):
            has_image_data = "image_data" in r and r["image_data"] is not None
            if has_image_data:
                input_ids = torch.tensor(r["prompts"], device=device)
                attention_mask = r["attention_mask"]
                all_multi_modal_inputs.append(r["image_data"])
            else:
                input_ids = torch.tensor(r["prompts"], device=device)
                attention_mask = r["attention_mask"]
                all_multi_modal_inputs.append(None)
            
            response_ids = torch.tensor(r["responses"], device=device)
            length = min(len(r["responses"]), response_len)
            responses[i, :length] = response_ids[:length]
            log_probs[i, :length] = torch.tensor(r["response_log_probs"][:length], device=device)
            per_sample_resp_loss_masks.append(r["response_loss_mask"][:length])
            
            # truncate prompt
            if len(input_ids) > prompt_len:
                print(f"======over long prompt in make_batch=======", len(input_ids))
                input_ids = input_ids[:prompt_len]
                attention_mask = attention_mask[:prompt_len]
            prompts_ids[i, -len(input_ids):] = input_ids

            if "reward" in r:
                rewards[i] = r["reward"]

            for k, v in r["obs_metrics"].items():
                obs_metrics[k][i] = v
        
        
        all_input_ids = torch.cat([prompts_ids, responses], dim=1)
        attention_mask = (all_input_ids != pad).int()
        
        # position_ids = torch.zeros((batch_size, 3, all_input_ids.shape[1]), dtype=all_input_ids.dtype, device=device)
        position_ids = torch.zeros((batch_size, 4, all_input_ids.shape[1]), dtype=all_input_ids.dtype, device=device)
        # position_ids = torch.zeros_like(all_input_ids, dtype=all_input_ids.dtype, device=device)
        for i in range(batch_size):
            if has_image_data and self.processor.image_processor.__class__.__name__ in ["Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast"]:
                try:
                    from verl.models.transformers.qwen2_vl import get_rope_index
                    pos_ids_3d = get_rope_index(
                        self.processor,
                        input_ids=all_input_ids[i],
                        image_grid_thw=all_multi_modal_inputs[i].get("image_grid_thw"),
                        attention_mask=attention_mask[i],
                    )
                    text_position_ids = torch.cumsum(attention_mask[i], dim=0) - 1
                    position_ids[i] = torch.cat([text_position_ids.view(1, -1), pos_ids_3d.squeeze(0)], dim=0)
                except Exception:
                    # import traceback
                    # traceback.print_exc()
                    position_ids[i] = torch.cumsum(attention_mask[i], dim=0) - 1
            elif self.processor.image_processor.__class__.__name__ == "Glm4vImageProcessor":
                # position_ids[i] = torch.cumsum(attention_mask[i], dim=0) - 1
                
                from verl.models.transformers.glm4_1_vl import get_rope_index
                pos_ids_3d, mrope_position_deltas = get_rope_index(
                        self.processor,
                        input_ids=all_input_ids[i].view(1, -1),
                        image_grid_thw=all_multi_modal_inputs[i].get("image_grid_thw"),
                        attention_mask=attention_mask[i].view(1, -1),
                    )
                text_position_ids = torch.cumsum(attention_mask[i], dim=0) - 1
                position_ids[i] = torch.cat([text_position_ids.view(1, -1), pos_ids_3d.squeeze(1)], dim=0)
                # print("pos shape", position_ids[i].shape)
            elif has_image_data and self.processor.image_processor.__class__.__name__ in ["Qwen2VLLMImageProcessor"]:
                pass
            else:
                position_ids[i] = torch.cumsum(attention_mask[i], dim=0) - 1

        full_loss_mask = torch.zeros((batch_size, prompt_len + response_len), dtype=torch.int, device=device)
        for i in range(batch_size):
            if i >= len(per_sample_resp_loss_masks) and val:
                continue
            resp_mask = torch.tensor(per_sample_resp_loss_masks[i], device=device)
            l = resp_mask.size(0)
            full_loss_mask[i, prompt_len:prompt_len + l] = resp_mask

        obs_keys = list(obs_metrics.keys())

        batch_dict = {
            "prompts": prompts_ids,
            "responses": responses,
            "input_ids": all_input_ids,
            "loss_mask": full_loss_mask,
            # "old_log_probs": log_probs,
            "behavior_log_probs": log_probs,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            #**obs_metrics,
            **({"rm_final_scores": rewards}),
        }
        # pad multi_modal_inputs
        if val and len(all_multi_modal_inputs) != batch_size:
            all_multi_modal_inputs = all_multi_modal_inputs + all_multi_modal_inputs[:batch_size - len(all_multi_modal_inputs)]
        non_tensor_batch_dict = {
            "multi_modal_inputs": all_multi_modal_inputs,
        }
        non_tensor_batch_dict = {key: np.array(value, dtype=object) for key, value in non_tensor_batch_dict.items()}
        batch = TensorDict(batch_dict, batch_size=batch_size)
        protocol = DataProto(batch=batch, non_tensor_batch=non_tensor_batch_dict, meta_info={"obs_keys": obs_keys})
        if dapo_metrics:
            protocol.meta_info["dapo_metrics"] = dapo_metrics
        del results
        return protocol