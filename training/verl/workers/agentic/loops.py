import asyncio
import json
import traceback
from typing import Awaitable, Callable, Any

from transformers import PreTrainedTokenizerBase, ProcessorMixin
from qwen_vl_utils import process_vision_info
import torch
import ray
import copy


SessionIdType = int
StarFnType = Callable[[int], Awaitable[dict]]
GenFnType = Callable[[Any], Awaitable]
ObsFnType = Callable[[Any, SessionIdType], Awaitable[dict]]
EndFnType = Callable[[int, bool], Awaitable]


def collect_metrics(src, tgt):
    for k, v in src.items():
        if k == "score":
            continue
        if k not in tgt:
            tgt[k] = v
        else:
            tgt[k] += v



def format_history(history, is_aw, max_pixels=500000, rollout_trace_path=''):
    """保持与 openai_chat_agent_loop 中相同的格式化规则。"""
    
    action_performed_so_far = ""
    steps = 0
    for msg in history:
        if msg['role'] == 'assistant':
            if '</think>' in msg["content"]:
                action_performed_so_far += f'Step {steps+1}: {msg["content"].split("</think>")[-1].strip()}\n'
            else:
                action_performed_so_far += f'Step {steps+1}: {msg["content"]}\n'
            steps += 1
    
    formatted_history = []
    for idx, msg in enumerate(history):
        content = msg.get("content")
        if isinstance(content, str):
            if idx == 0:
                assert msg['role'] == 'system', f'{msg=}'
            msg['content'] = msg['content'].replace('<ans>', '<answer>').replace('</ans>', '</answer>')
            if idx == 1 and is_aw:
                assert history[idx-1]['role'] == 'system', f'{history[idx-1]=}'
                if not msg['content'].startswith('[AndroidWorld Benchmark]'):
                    msg['content'] = '[AndroidWorld Benchmark]' + msg['content']

                    # print(f"======msg['content'] {idx}", msg['content'])
            formatted_history.append(msg)
        else:
            # ChatML multimodal message
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url" and isinstance(part["image_url"], dict):
                    new_parts.append({"type": "image","image_url":part["image_url"]["url"], "max_pixels": max_pixels})
                elif isinstance(part, dict) and part.get("type") == "text":
                    part['text'] = part['text'].replace('<ans>', '<answer>').replace('</ans>', '</answer>')
                    if idx == 1 and is_aw:
                        assert history[idx-1]['role'] == 'system', f'{history[idx-1]=}'
                        if not part['text'].startswith('[AndroidWorld Benchmark]'):
                            part['text'] = '[AndroidWorld Benchmark]' + part['text']
                    new_parts.append(part)
            formatted_history.append({"role": msg["role"], "content": new_parts})
    
    return formatted_history


def format_history_print(history):
    """保持与 openai_chat_agent_loop 中相同的格式化规则。"""
    formatted_history = []
    for msg in history:
        content = msg.get("content")
        if isinstance(content, str):
            formatted_history.append(msg)
        else:
            # ChatML multimodal message
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url" and isinstance(part["image_url"], dict):
                    new_parts.append({"type": "image_url","image_url": part["image_url"]['url'][:100]})
                elif isinstance(part, dict) and part.get("type") == "image_url" and isinstance(part["image_url"], str):
                    new_parts.append({"type": "image_url","image_url": part["image_url"][:100]})
                else:
                    new_parts.append(part)
            formatted_history.append({"role": msg["role"], "content": new_parts})
    return formatted_history


def add_reward(all_turns_data):
    final_reward = all_turns_data[-1]["reward"]
    return_data = []
    for turn_data in all_turns_data:
        turn_data["reward"] = final_reward
        turn_data["done"] = False
        return_data.append(turn_data)
    return_data[-1]["done"] = True
    print("===== final_reward =======", final_reward)
    return return_data


async def openai_chat_agent_loop(
    start_args: dict,
    start_fn: StarFnType,
    gen_fn: GenFnType,
    obs_fn: ObsFnType,
    end_fn: EndFnType,
    max_turns: int,
    # max_length: int,
    prompt_length: int,
    response_length: int,
    tokenizer: PreTrainedTokenizerBase,
    processor: ProcessorMixin,
    tool_call_parser: str,
    rollout_trace_path: str,
    image_only: bool,
    incomplete_punishment: float = 0,
    max_pixels: int = 500000,
    val: bool = False,
    **_
) -> dict:
    max_length = prompt_length + response_length
    done = False
    reward = 0
    status = ""
    obs_metrics = {}
    all_turns_data = []
    save_turns_data = []

    # start
    start = await start_fn(**start_args)
    history = start.pop("messages")
    tools = start.pop("tools", None)
    sid = start.pop("sid")
    collect_metrics(start.get("metrics", {}), obs_metrics)
    
    rollout_trace_path = f"{rollout_trace_path}/val" if val else f"{rollout_trace_path}/train"
    processed_start_args = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in start_args.items()}
    # with open(f"{rollout_trace_path}/{sid}.json", "w") as f:
    #     json.dump({**processed_start_args, "result": -1, "status": status, "save_turns_data": save_turns_data}, f, indent=2, ensure_ascii=False)
    
    start_args["sid"] = sid

    # image_data = start.pop("image_data", None)
    #history, image_data = format_history(history, image_data)

    is_aw = 'android_world' in start_args['name']
    history = format_history(history, is_aw, max_pixels=max_pixels, rollout_trace_path=rollout_trace_path)
   

    # if image_data:
    #     for idx, hist in enumerate(history):
    #         if isinstance(hist['content'], str):
    #             continue
    #         if isinstance(hist['content'], list):
    #             for idx_2, hist_content in enumerate(hist['content']):
    #                 if hist_content['type'] == 'image' and hist_content.get('image') is None:
    #                     history[idx]['content'][idx_2]['image'] = image_data.pop(0)
    #     assert len(image_data) == 0, "length not match for image_data"
    try:
        image_inputs, video_inputs = process_vision_info(history)
        prompt_text = tokenizer.apply_chat_template(history, tools=tools, tokenize=False, add_generation_prompt=True)
        if 'point' in start_args['name']:
            prompt_text = prompt_text + '<think>\n'
        prompt_ids = processor.tokenizer.encode(prompt_text, add_special_tokens=False)
        inputs = processor(
                images=image_inputs,
                text=[prompt_text],
                add_special_tokens=False,
                return_tensors="pt",
            )
        input_ids = inputs.pop("input_ids")[0].tolist()
        attention_mask = inputs.pop("attention_mask")[0]
        print("====== image_inputs =======", image_inputs)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("======history=======", format_history_print(history))
        #print("======image_data=======", image_data)
        # print("======tokenizer=======", tokenizer)
        raise e
    
    # interact
    for turn in range(max_turns):
        # print("======= history =======", format_history_print(history))
        image_token_indices = [i for i, token_id in enumerate(input_ids) if token_id == processor.image_token_id]
        rightmost_image_token_index = max(image_token_indices) if image_token_indices else -1
        if len(input_ids) >= prompt_length and rightmost_image_token_index+1 >= prompt_length:
            print(f"====== over long prompt and image token indices larger than prompt length =======: {len(input_ids)}, the trace will be dropped.")
            await end_fn(sid, False)
            return []
        
        text, log_probs = await gen_fn(prompt_text, image_inputs, val) # await gen_fn(prompt_ids, image_inputs, val)
        
        # print("======= text =======", text)
        new_ids = [t[1] for t in log_probs]
        new_log_probs = [t[0] for t in log_probs]
        # ids += new_ids
        current_turn_data = {
            "prompts": input_ids,
            "attention_mask": attention_mask,
            "responses": new_ids,
            "response_loss_mask": [1] * len(new_ids),
            "response_log_probs": new_log_probs,
            "reward": 0,  # 暂时为0，后面会更新
            "obs_metrics": {},
            "image_data": {"pixel_values": ray.put(inputs["pixel_values"]), "image_grid_thw": inputs["image_grid_thw"]} if image_inputs is not None else None,
            "sid": sid,
        }
        
        save_turns_data.append({
            "turn_idx": turn,
            "response": text,
            "messages": copy.deepcopy(history),
        })

        message: dict[str, str | list] = {
            "role": "assistant",
        }
        # text = text.replace('<answer>', '<ans>').replace('</answer>', '</ans>')
        if 'point' in start_args['name']:
            text = '<think>\n' + text
        if tools:
            from sglang.srt.function_call.function_call_parser import FunctionCallParser
            from sglang.srt.openai_api.protocol import Tool
            parser = FunctionCallParser(tools=[Tool.model_validate(tool) for tool in tools], tool_call_parser=tool_call_parser)
            try:
                normal_text, info_list = parser.parse_non_stream(text)
            except:
                normal_text = text
                info_list = []
            message["content"] = normal_text
            message["tool_calls"] = [{
                "id": str(info.tool_index),
                "function": {
                    "name": info.name,
                    "arguments": info.parameters,
                }
            } for info in info_list]
        else:
            message["content"] = text

        # history.append(message)

        obs = await obs_fn(message, sid)
        try:
            messages = obs.pop("messages")
            done = obs.pop("finish")
            reward = obs.pop("reward")
            status = obs.pop("status")
            # history.extend(messages)
            history = format_history(messages, is_aw, max_pixels=max_pixels, rollout_trace_path=rollout_trace_path)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("======obs=======", obs)
        
        collect_metrics(obs.get("metrics", {}), obs_metrics)
        
        current_turn_data["reward"] = reward
        current_turn_data["obs_metrics"] = obs_metrics
        if val:
            ray.get(current_turn_data["image_data"]["pixel_values"])
            del current_turn_data["image_data"]
        all_turns_data.append(current_turn_data)


        if done:
            break
        
        try:
            # prompt_ids = tokenizer.apply_chat_template(history, tools=tools, tokenize=True, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(history)
            prompt_text = tokenizer.apply_chat_template(history, tools=tools, tokenize=False, add_generation_prompt=True)
            if 'point' in start_args['name']:
                prompt_text = prompt_text + '<think>\n'
            prompt_ids = processor.tokenizer.encode(prompt_text, add_special_tokens=False)
            inputs = processor(
                    images=image_inputs,
                    text=[prompt_text],
                    add_special_tokens=False,
                    return_tensors="pt",
                )
            input_ids = inputs.pop("input_ids")[0].tolist()
            attention_mask = inputs.pop("attention_mask")[0]
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("======history=======", format_history_print(history))
            raise e
        
        # 更新当前ids为新的prompt
        current_ids = list(prompt_ids)

        if done or len(current_ids) >= max_length:
            break

    await end_fn(sid, done)
    
    # if status == "task error":
    #     with open(f"{rollout_trace_path}/{sid}.json", "w") as f:
    #         json.dump({**processed_start_args, "result": -1, "status": status}, f, indent=2, ensure_ascii=False) # , "save_turns_data": save_turns_data
    #     print(f"Sid: {sid} got {status} when rollout samples....")
    #     return []
    # else:
    #     if all_turns_data[-1]["reward"] >= 0.8:
    #         with open(f"{rollout_trace_path}/{sid}.json", "w") as f:
    #             json.dump({**processed_start_args, "result": all_turns_data[-1]["reward"], "status": status, "save_turns_data": save_turns_data}, f, indent=2, ensure_ascii=False)
    #     else:
    #         with open(f"{rollout_trace_path}/{sid}.json", "w") as f:
    #             json.dump({**processed_start_args, "result": all_turns_data[-1]["reward"], "status": status, "save_turns_data": []}, f, indent=2, ensure_ascii=False)
                
    for turn_data in all_turns_data:
        if len(turn_data["responses"]) > response_length:
            print("======over long response=======", len(turn_data["responses"]))
            turn_data["responses"] = turn_data["responses"][:response_length]
            turn_data["response_loss_mask"] = turn_data["response_loss_mask"][:response_length]
            turn_data["response_log_probs"] = turn_data["response_log_probs"][:response_length]

    all_turns_data = add_reward(all_turns_data)
    # f"{tokenizer.decode(turn_data['prompts'])=}"
    '''
    print("-"*100)
    if torch.distributed.get_rank() == 0:
        for idx, turn_data in enumerate(all_turns_data):
            print(f"===============turn_data['prompts'] {idx}", tokenizer.decode(turn_data['prompts']))
            print(f"===============turn_data['responses'] {idx}", tokenizer.decode(turn_data['responses']))
            print(f"===============turn_data['response_loss_mask'] {idx}", tokenizer.decode(turn_data['response_loss_mask']))
            print(f"===============turn_data['reward'] {idx}", turn_data['reward'])'''
    #print("=============len(all_turns_data)", len(all_turns_data))

    return all_turns_data


async def retry_openai_chat_agent_loop(
    start_args: dict,
    start_fn: StarFnType,
    gen_fn: GenFnType,
    obs_fn: ObsFnType,
    end_fn: EndFnType,
    max_turns: int,
    # max_length: int,
    prompt_length: int,
    response_length: int,
    tokenizer: PreTrainedTokenizerBase,
    processor: ProcessorMixin,
    tool_call_parser: str,
    rollout_trace_path: str,
    image_only: bool,
    incomplete_punishment: float = 0,
    val: bool=False,
    max_pixels: int = 500000,
    retry_times: int = 5,
    **_
) -> dict | None:
    for i in range(retry_times):
        try:
            return await openai_chat_agent_loop(
                start_args,
                start_fn,
                gen_fn,
                obs_fn,
                end_fn,
                max_turns,
                prompt_length,
                response_length,
                tokenizer,
                processor,
                tool_call_parser,
                rollout_trace_path,
                image_only,
                incomplete_punishment,
                max_pixels,
                val,
            )
        except:
            traceback.print_exc()
            print(f"nodedup Retrying openai_chat_agent_loop... {start_args=}")
            await asyncio.sleep(1)
            if start_args.get("sid"):
                await end_fn(start_args["sid"], False)
                print(f"ended sid: {start_args['sid']}")
    print(f"nodedup Failed to run openai_chat_agent_loop after retries! {start_args=}")
    return []
