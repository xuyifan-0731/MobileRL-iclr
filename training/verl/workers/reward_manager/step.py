from collections import defaultdict
from typing import Dict, List
import enum

import torch
from verl import DataProto
from verl.utils.reward_score import _default_compute_score

__all__ = [
    "check_uid_uuid_reward",
    "StepRewardManager",
]

################################################################################
# Helper: UID / UUID integrity check + dynamic per‑UID length‑penalty           #
################################################################################

def check_uid_uuid_reward(
    uids: List[str],
    uuids: List[str],
    scores: torch.Tensor,
    *,
    verbose: bool = False,
) -> Dict[str, float]:
    """Validate (uid, uuid, reward) triples & compute *per‑uuid* success‑length penalty."""
    uuid_to_uid: Dict[str, str] = {}
    uid_conflict = []
    for uid, uuid in zip(uids, uuids):
        if uuid in uuid_to_uid and uuid_to_uid[uuid] != uid:
            uid_conflict.append((uuid, uuid_to_uid[uuid], uid))
        uuid_to_uid.setdefault(uuid, uid)

    uuid_values = defaultdict(set)
    reward_conflict = []
    for uuid, r in zip(uuids, scores.tolist()):
        uuid_values[uuid].add(r)
        if len(uuid_values[uuid]) > 1:
            reward_conflict.append(uuid)

    uid_to_uuid_rewards: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for uid, uuid, r in zip(uids, uuids, scores.tolist()):
        uid_to_uuid_rewards[uid][uuid].append(r)

    uuid_penalty: Dict[str, float] = {}
    for uid, uuid_dict in uid_to_uuid_rewards.items():
        success_lens = [len(rs) for rs in uuid_dict.values() if any(r == 1 for r in rs)]
        has_success = bool(success_lens)
        min_len = min(success_lens) if has_success else 0
        for uuid, rs in uuid_dict.items():
            l = len(rs)
            if all(r == 0 for r in rs):
                p = 0.0
            elif has_success:
                p = max(l - min_len, 0) / (2 * l)
            else:
                p = 0.0
            uuid_penalty[uuid] = float(p)

    if verbose:
        if uid_conflict:
            print("❗ UID/UUID 冲突:")
            for uuid, u1, u2 in uid_conflict:
                print(f"  {uuid}: {u1} vs {u2}")
        if reward_conflict:
            print("❗ Reward 冲突 (同 uuid 不同值):", set(reward_conflict))
    return uuid_penalty

################################################################################
# StepRewardManager                                                          #
################################################################################

class NormType(str, enum.Enum):
    TOKEN = "token"   # 1/L_token
    STEP  = "step"    # 1/L_step  (each trajectory step counts once)

class StepRewardManager:
    """Reward generator with independent flags **and** two normalisation modes.

    Parameters
    ----------
    length_normalise : bool | str | NormType
        * ``False`` → no length norm.
        * ``"token"`` (default) → distribute reward uniformly to every *token*.
        * ``"step"``           → distribute reward to last token of each *step*,
          scaled by 1 / (#steps for that uuid in current batch).
    success_penalty : bool
        Whether to subtract per‑uuid shortest‑success penalty.
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        length_normalise: bool | str | NormType = NormType.TOKEN,
        success_penalty: bool = True,
        reward_fn_key: str = "data_source",
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

        if isinstance(length_normalise, bool):
            self._length_norm_mode_default = NormType.TOKEN if length_normalise else None
        else:
            self._length_norm_mode_default = NormType(length_normalise)
        self._penalty_flag_default = bool(success_penalty)

    def __call__(self, data: DataProto, return_dict: bool=False, val: bool=False):
        """Return reward tensor matching ``responses`` shape."""
        norm_mode = None if val else self._length_norm_mode_default
        enable_penalty = (not val) and self._penalty_flag_default

        if "rm_scores" in data.batch:
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        if "rm_final_scores" not in data.batch:
            raise KeyError("DataProto.batch 缺少 'rm_final_scores' 字段")

        # step‑count per uuid (用于 STEP 归一)
        if norm_mode == NormType.STEP:
            step_counts = defaultdict(int)
            for uuid in data.non_tensor_batch["uuid"]:
                step_counts[uuid] += 1
        else:
            step_counts = defaultdict(lambda: 1)  # type: ignore

        # penalty
        if enable_penalty:
            try:
                penalty_table = check_uid_uuid_reward(
                    data.non_tensor_batch["uid"],
                    data.non_tensor_batch["uuid"],
                    data.batch["rm_final_scores"],
                    verbose=False,
                )
            except Exception as exc:
                print("[RewardMgr] penalty_calc fail:", exc)
                penalty_table = defaultdict(lambda: 0.0)  # type: ignore
        else:
            penalty_table = defaultdict(lambda: 0.0)  # type: ignore

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        for i in range(len(data)):
            item = data.batch[i]
            prompt_len = item["prompts"].shape[-1]
            resp_len = int(item["attention_mask"][prompt_len:].sum())
            if resp_len == 0:
                continue

            raw_score = float(item["rm_final_scores"])
            if enable_penalty:
                uuid_i = data.non_tensor_batch["uuid"][i]
                penalty = float(penalty_table[uuid_i])
            else:
                penalty = 0.0
            final_score = raw_score - penalty

            if norm_mode == NormType.TOKEN:
                per_token = final_score / resp_len
                reward_tensor[i, prompt_len : prompt_len + resp_len] = per_token
            elif norm_mode == NormType.STEP:
                # Step‑level: same per‑token reward across the whole response,
                # scaled by 1 / (#steps for this uuid in current mini‑batch).
                uuid_i = data.non_tensor_batch["uuid"][i]
                per_token = final_score / step_counts[uuid_i]
                reward_tensor[i, prompt_len : prompt_len + resp_len] = per_token
            else:  # no normalisation:  # no normalisation
                reward_tensor[i, resp_len - 1] = final_score

        if return_dict:
            return {"reward_tensor": reward_tensor}
        else:
            return reward_tensor



        already_print_data_sources = {}

        # for i in range(len(data)):
        #     data_item = data[i]  # DataProtoItem

        #     prompt_ids = data_item.batch['prompts']

        #     prompt_length = prompt_ids.shape[-1]

        #     valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        #     valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        #     response_ids = data_item.batch['responses']
        #     valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        #     # valid_response_ids = response_ids[:valid_response_length]
        #     valid_response_ids = response_ids

        #     # decode
        #     sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        #     sequences_str = self.tokenizer.decode(sequences)

        #     prompt_str = self.tokenizer.decode(valid_prompt_ids)
        #     ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

        #     data_source = data_item.non_tensor_batch['data_source']

        #     extra_info = data_item.non_tensor_batch.get('extra_info', None)

        #     score = self.compute_score(
        #         data_source=data_source,
        #         solution_str=sequences_str,
        #         ground_truth=ground_truth,
        #         extra_info=extra_info,
        #         question=prompt_str
        #     )
        #     reward_tensor[i, valid_response_length - 1] = score

        #     if data_source not in already_print_data_sources:
        #         already_print_data_sources[data_source] = 0

        #     if already_print_data_sources[data_source] < self.num_examine:
        #         already_print_data_sources[data_source] += 1
        #         # print(sequences_str)

        # parallel eval

        import concurrent.futures

        def compute_score_for_item(i, data_item):
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # print(f"naive reward manager {self.tokenizer=}")
            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                question=prompt_str,
                tokenizer=self.tokenizer
            )
            return i, valid_response_length, score, data_source, sequences_str

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(data)) as executor:
            futures = [executor.submit(compute_score_for_item, i, data[i]) for i in range(len(data))]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            for i, valid_response_length, score, data_source, sequences_str in results:
                reward_tensor[i, valid_response_length - 1] = score

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    # print(sequences_str)

        return reward_tensor