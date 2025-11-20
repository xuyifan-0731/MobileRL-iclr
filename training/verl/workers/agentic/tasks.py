import asyncio
import random
import traceback

import aiohttp
import torch


async def dummy(*_, **__):
    pass


async def openai_chat_start(index, name, url, sid=None):
    print(f"starting session {index=}")
    if isinstance(index, torch.Tensor):
        index = index.item()
    # avoid peak
    await asyncio.sleep(random.randint(0, 3))

    max_retries = 5
    for i in range(max_retries):
        try:
            ret = None
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), trust_env=True) as session:
                async with session.post(url + "/start_sample", json={
                    "index": index,
                    "name": name,
                }) as response:
                    ret = await response.json()
                    response.raise_for_status()
                    ret["sid"] = response.headers["session_id"]
                    assert ret["messages"], f"Empty messages in response {ret['sid']=}"
                    return ret
        except Exception:
            print(f"API call failed (attempt {i+1}/{max_retries}): {ret=}")
            traceback.print_exc()
            if i < max_retries - 1:
                await asyncio.sleep(random.randint(1, 10))
                continue
            raise
    raise NotImplementedError("Unreachable code")


async def openai_chat_obs(message, sid, url, **_):
    payload = {"messages": [message]}
    header = {"session_id": str(sid)}
    metrics = {"client_observation_times": 1}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), trust_env=True) as session:
        async with session.post(url + "/interact", json=payload, headers=header) as response:
            assert response.status == 200, f"Wrong status code: {sid=} {response.status=} {await response.text()}"
            ret = await response.json()
    ret["metrics"] = ret.get("metrics", {}) | metrics
    return ret


async def openai_chat_end(sid, done, url):
    if done:
        return
    header = {"session_id": sid}
    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(url + "/cancel", headers=header, json={}):
                pass
    except Exception as e:
        print(f"API call failed when ending: {e}")
