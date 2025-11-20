import torch
from collections import defaultdict
from typing import Dict, List, Tuple
import math

    
def trace_level_kl_clip(batch, kl_threshold: float = 0.5):
    """Mute samples whose token‑level KL divergence exceeds ``kl_threshold`` **and**
    whose reward is non‑positive by zeroing their ``response_mask`` (and
    ``loss_mask`` if present) *in‑place*.

    Unlike the previous version, this implementation **does not create a copy**
    of the batch.  The original input ``batch`` is modified directly so it no
    longer contributes to the PPO loss, while the batch shape, gradient
    accumulation schedule, and any statistics dependent on the number of
    samples stay intact.

    Parameters
    ----------
    batch : DataProto
        Mini‑batch produced by the data pipeline.  It must contain the
        following tensors:

        * ``old_log_probs``  : (N, L_sub)
        * ``ref_log_prob``   : (N, L_sub)
        * ``rm_final_scores``: (N,)
        * ``attention_mask`` : (N, L_full) – only used for token alignment.
        * ``response_mask``  : (N, L_sub) – will be zeroed for muted rows.
        * ``loss_mask``      : (optional, N, L_sub) – also zeroed if present.

    kl_threshold : float, default ``0.5``
        A token whose KL > ``kl_threshold`` flags its row for muting.

    Returns
    -------
    high_kl_batch : DataProto
        Slice containing the rows that were muted (useful for logging).
    new_batch : DataProto
        The **same object** as *batch*, now containing modified masks.
    """

    # ---------------------------------------------------------------------
    # 1.  Handles & tensors
    # ---------------------------------------------------------------------
    attn_mask    = batch.batch["attention_mask"]         # (N, L_full)
    log_prob     = batch.batch["old_log_probs"]          # (N, L_sub)
    ref_log_prob = batch.batch["ref_log_prob"]           # (N, L_sub)
    rm_scores    = batch.batch["rm_final_scores"]        # (N,)

    # ---------------------------------------------------------------------
    # 2.  Identify rows with high‑KL tokens and non‑positive reward
    # ---------------------------------------------------------------------
    bad_reward_mask = rm_scores <= 0                      # (N,)

    if bad_reward_mask.any():
        L_sub = log_prob.size(1)
        sub_mask = attn_mask[..., -L_sub:].bool()         # align with log_prob

        kl_token = log_prob - ref_log_prob               # (N, L_sub)
        mean_kl = (kl_token * sub_mask).sum(dim=1) / sub_mask.sum(dim=1)  # (N,)
        has_high_kl = bad_reward_mask & (mean_kl > kl_threshold)   # (N,)
    else:
        has_high_kl = torch.zeros_like(rm_scores, dtype=torch.bool)

    high_kl_mask = has_high_kl                            # (N,)

    # ---------------------------------------------------------------------
    # 3.  Mute offending rows in‑place
    # ---------------------------------------------------------------------
    if high_kl_mask.any():
        if "loss_mask" in batch.batch:
            batch.batch["loss_mask"][high_kl_mask] = 0
        else:
            raise ValueError("loss_mask not in batch")
    # Return the same (now muted) batch as new_batch for interface consistency
    return batch



def mute_uid_all_zero_reward(
    batch,
) -> Tuple["DataProto", "DataProto"]:
    """
    将 **所有 rm_final_scores 均为 0 的 uid** 对应样本静音：
    把这些样本的 ``loss_mask``（以及可选的 ``response_mask``）全置 0。
    
    同时限制每组数据中负例最多为正例的2倍。
    
    返回 (muted_batch, batch)
    -----------------------------------------------
    "muted_batch" 只是切片，方便你做日志 / 可视化；
    "batch" 与输入对象完全相同，只是已原地修改。
    """
    # ------------------ 1. 句柄 ------------------
    uids        = batch.non_tensor_batch["uid"]           # List[str], len=N
    rm_scores   = batch.batch["rm_final_scores"]          # (N,)
    loss_mask   = batch.batch["loss_mask"]                # (N, L_sub)
    resp_mask   = batch.batch.get("response_mask", None)  # (N, L_sub) 或 None

    device = rm_scores.device
    N      = rm_scores.size(0)

    # ------------------ 2. 收集每个 uid 的行索引 ------------------
    uid_to_rows = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid_to_rows[uid].append(idx)

    # ------------------ 3. 找出 "全 0 奖励" 的 uid 并处理负例比例 ------------------
    mute_row_idx = []
    mute_uids    = []           # ### DEBUG
    
    for uid, rows in uid_to_rows.items():
        uid_scores = rm_scores[rows]
        
        # 原有逻辑：如果所有分数都是0，则静音该uid的所有样本
        if (uid_scores == 0).all():
            mute_row_idx.extend(rows)
            mute_uids.append(uid)
        else:
            # 新增逻辑：限制负例数量不超过正例的2倍
            positive_indices = [rows[i] for i in range(len(rows)) if uid_scores[i] > 0]
            negative_indices = [rows[i] for i in range(len(rows)) if uid_scores[i] <= 0]
            
            num_positive = len(positive_indices)
            num_negative = len(negative_indices)
            
            # 如果负例数量超过正例的2倍，随机选择一些负例进行静音
            if num_negative > 2 * num_positive:
                max_negative = 2 * num_positive
                num_to_mute = num_negative - max_negative
                
                # 使用torch.randperm随机选择需要静音的负例
                perm = torch.randperm(len(negative_indices))[:num_to_mute]
                mute_negative_indices = [negative_indices[i] for i in perm.tolist()]
                mute_row_idx.extend(mute_negative_indices)
                
                print(f"[mute_uid_all_zero_reward] uid {uid}: positive={num_positive}, negative={num_negative}, "
                      f"muting {num_to_mute} negative samples to maintain 2:1 ratio")

    # ------------------ 4. 原地静音并打印调试信息 ------------------
    if mute_row_idx:                                     # 至少有需要静音的行
        mute_mask = torch.zeros(N, dtype=torch.bool, device=device)
        mute_mask[mute_row_idx] = True

        # ### DEBUG: 打印静音前的 loss_mask/resp_mask（可视化首行）
        if mute_row_idx:
            print("[mute_uid_all_zero_reward] loss_mask BEFORE mute (first row):",
                  loss_mask[mute_row_idx[0]].clone())

        # 真正归零
        loss_mask[mute_mask] = 0
        if resp_mask is not None:
            resp_mask[mute_mask] = 0

        # ### DEBUG: 打印静音后的 loss_mask（验证已归零）
        if mute_row_idx:
            print("[mute_uid_all_zero_reward] loss_mask AFTER  mute (first row):",
                  loss_mask[mute_row_idx[0]])
    else:
        print("[mute_uid_all_zero_reward] no uid has all-zero reward and no negative ratio adjustment needed.")

    return batch
