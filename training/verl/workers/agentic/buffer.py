import asyncio
from collections import deque, defaultdict
from math import lcm

from verl import DataProto


def dapo_metrics(uid_scores, uid_turns):
    uuid_scores = []
    for uid, scores in uid_scores.items():
        uuid_scores.extend(scores)
    
    ret = {
        "traj_reward": sum(uuid_scores) / len(uuid_scores),
        "BON_reward": sum([max(scores) for scores in uid_scores.values()]) / len(uid_scores),
        "average_margin": sum(max(scores) - sum(scores) / len(scores) for scores in uid_scores.values()) / len(uid_scores),
        "average_turns": sum([turn for uid in uid_turns for turn in uid_turns[uid]]) / len(uuid_scores),
        "num_uids": len(uid_scores),
        "effective_uids": sum([len(set(scores)) > 1 for scores in uid_scores.values()]) / len(uid_scores),
        "all_1_uids": sum([len(set(scores)) == 1 and scores[0] == 1 for scores in uid_scores.values()]) / len(uid_scores),
        "all_leq0_uids": sum([len(set(scores)) == 1 and scores[0] <= 0 for scores in uid_scores.values()]) / len(uid_scores),
    }
    
    dapo_metrics = {f"dapo/{k}": v for k, v in ret.items()}
    return dapo_metrics

class Buffer:
    def __init__(self, max_size, group_size, dapo_sampling=False):
        self.max_size = max_size
        self.strict_group = group_size > 1
        self.group_size = group_size
        self.groups = {}
        self.put_signal = asyncio.Event()
        self.get_signal = asyncio.Event()
        self.queue = deque()
        self.dapo_sampling = dapo_sampling
        self.uid_scores = defaultdict(list)
        self.uid_turns = defaultdict(list)

    async def add(self, result, other: DataProto):
        while len(self.queue) >= self.max_size:
            await self.get_signal.wait()
            self.get_signal.clear()
        if self.strict_group:
            uid = other.non_tensor_batch["uid"]
            if uid not in self.groups:
                self.groups[uid] = []
            self.groups[uid].append((result, other))
            if len(self.groups[uid]) >= self.group_size:
                if self.dapo_sampling:
                    rewards = [result[0]['reward'] for result, other in self.groups[uid]]
                    if any(r == 1 for r in rewards) and not all(r == 1 for r in rewards):
                        self.queue.extend(self.groups[uid])
                    self.uid_scores[uid].append(rewards)
                    self.uid_turns[uid].append([len(result) for result, other in self.groups[uid]])
                else:
                    self.queue.extend(self.groups[uid])
                self.put_signal.set()
                del self.groups[uid]
        else:
            self.queue.append((result, other))
            self.put_signal.set()

    async def get(self, minimum, multiple: int = 1):
        if self.strict_group:
            multiple = lcm(self.group_size, multiple)
        while True:
            # length = len(self.queue) // multiple * multiple
            length = len(self.queue)
            if length >= minimum:
                items = [self.queue.popleft() for _ in range(length)]
                if self.dapo_sampling:
                    dapo_metrics = dapo_metrics(self.uid_scores, self.uid_turns)
                    for result, other in items:
                        for turn in result:
                            turn['dapo_metrics'] = dapo_metrics

                # if self.dapo_sampling:
                #     uid_traces = defaultdict(list)
                #     for result, other in items:
                #         uid_traces[other.non_tensor_batch["uid"]].append((result, other))
                #     valid_traces = []
                #     for traces in uid_traces.values():
                #         rewards = [result[0]['reward'] for result, other in traces]
                #         if any(r == 1 for r in rewards) and not all(r == 1 for r in rewards):
                #             valid_traces.extend(traces)
                #     # length = len(valid_traces) // multiple * multiple
                #     length = len(valid_traces)
                #     if length >= minimum:
                #         items = valid_traces[:length]
                #         remaining = valid_traces[length:]
                #         self.queue.extendleft(reversed(remaining))
                #     else:
                #         self.queue.extendleft(reversed(valid_traces))
                #         print(f"not enough items for dapo: {length=} {self.group_size=} {multiple=} {self.strict_group=} {minimum=}")
                #         await self.put_signal.wait()
                #         self.put_signal.clear()
                #         continue
                    
                print(f"returned {len(items)=} remaining {len(self.queue)=}")
                self.get_signal.set()
                return items
            print(f"not enough items: {length=} {self.group_size=} {multiple=} {self.strict_group=} {minimum=}")
            await self.put_signal.wait()
            self.put_signal.clear()
