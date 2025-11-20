import torch
from collections import defaultdict
from typing import Dict, List
import math

def masked_row_mean(adv: torch.Tensor,
                    attn_mask: torch.Tensor,
                    eps: float = 1e-8,
                    keepdim: bool = False) -> torch.Tensor:
    """
    adv.shape  : (B, L_adv)
    attn_mask.shape : (B, L_full)

    自动把 attn_mask 裁剪为最后 L_adv 列后再求均值。
    """
    # 只取与 adv 对齐的部分
    L_adv   = adv.size(1)
    mask    = attn_mask[..., -L_adv:].float()            # (B, L_adv)

    masked_sum = (adv * mask).sum(dim=1)                 # (B,)
    valid_cnt  = mask.sum(dim=1).clamp_min(eps)          # (B,)
    mean       = masked_sum / valid_cnt                  # (B,)

    return mean.unsqueeze(-1) if keepdim else mean


def check_uid_uuid_adv(
    uids: List[str],
    uuids: List[str],
    advs: torch.Tensor,
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
    adv_conflict = []
    for uuid, r in zip(uuids, advs.tolist()):
        uuid_values[uuid].add(r)
        if len(uuid_values[uuid]) > 1:
            adv_conflict.append(uuid)

    uid_to_uuid_rewards: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for uid, uuid, r in zip(uids, uuids, advs.tolist()):
        uid_to_uuid_rewards[uid][uuid].append(r)

    # 

"""
Replay buffer that stores *traces* grouped by ``uuid`` and supports high‑advantage
replay sampling with balanced eviction.

Key ideas
---------
* **Per‑uuid insertion** – the buffer treats one *uuid* (a trajectory) as the
  atomic element. When we decide a uuid is worth keeping, *all* rows that belong
  to that uuid are stored together.
* **Top‑k by mean advantage** – on every ``push`` we look at the incoming
  mini‑batch, compute the row‑wise advantage mean, aggregate by uuid, and keep
  the uuids whose average advantage sits in the top ``top_ratio`` (default
  ``0.10``).
* **Balanced FIFO × advantage eviction** – the buffer is capped at
  ``max_size`` unique uuids. When it overflows we compute a *score* that equally
  balances **(a) normalised average advantage** and **(b) recency**; uuids with
  the *lowest* score are evicted first. ``balance_alpha`` controls the weight
  (``alpha=1`` → pure advantage, ``alpha=0`` → pure recency/FIFO).
* **High‑adv replay on sampling** – each call to ``sample_for_training`` returns
  a concatenated mini‑batch containing the highest‑advantage uuids from the
  current buffer. The size is ``ceil(replay_ratio * n_uuid)``.
* **Rich debug logs** – enable ``debug=True`` to print inserts, evictions and
  samples with their scores/advantages.

The implementation is framework‑agnostic: your *batch* just needs Boolean /
integer indexing and the usual keys (``uid``, ``uuid``, ``advantages``,
``attention_mask``).
"""

import math
import random
from collections import OrderedDict, defaultdict
from typing import Callable, Dict, List, Sequence

import torch


def _concat_batches(batches: Sequence):
    """Try to concatenate *batches* using the best method available."""
    if not batches:
        raise ValueError("batches must contain at least one element")
    first = batches[0]
    # Preferred explicit methods
    for method in ("cat", "concat"):
        fn = getattr(first, method, None)
        if fn is not None:
            return fn(batches)
    # Fallback to ``+``
    result = first
    for b in batches[1:]:
        result = result + b  # type: ignore[operator]
    return result


class ReplayBuffer:
    """Per‑uuid replay buffer with balanced eviction *and* age‑based purge.

    Parameters
    ----------
    max_size : int
        Max number of **unique** uuids kept simultaneously.
    top_ratio : float, default 0.10
        Fraction of uuids from each *incoming* batch that are inserted (chosen
        by highest average advantage in that batch).
    replay_ratio : float, default 0.10
        Fraction of the buffer returned by :py:meth:`sample_for_training`.
    balance_alpha : float, default 0.5
        Weight for *advantage* in the eviction score vs. *recency*.
    max_age_steps : int, default 4
        Automatic purge window. A uuid inserted more than this many *push* calls
        ago is dropped on the next push. Set ``None`` / ``<=0`` to disable.
    seed : int | None, default ``None``
        RNG seed for shuffling tie‑breakers.
    debug : bool, default ``False``
        Print detailed logs if ``True``.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        max_size: int = 512,
        *,
        top_ratio: float = 0.25,
        replay_ratio: float = 0.25,
        balance_alpha: float = 0.5,
        max_age_steps: int = 4,
        seed: int | None = None,
        debug: bool = False,
    ) -> None:
        if not (0.0 < top_ratio <= 1.0):
            raise ValueError("top_ratio must be in (0, 1]")
        if not (0.0 < replay_ratio <= 1.0):
            raise ValueError("replay_ratio must be in (0, 1]")
        if not (0.0 <= balance_alpha <= 1.0):
            raise ValueError("balance_alpha must be in [0, 1]")
        if max_size < 1:
            raise ValueError("max_size must be >= 1")

        self.max_size = max_size
        self.top_ratio = top_ratio
        self.replay_ratio = replay_ratio
        self.adv_mean_fn = masked_row_mean
        self.balance_alpha = balance_alpha
        self.max_age_steps = max_age_steps
        self.debug = debug
        self.rng = random.Random(seed)

        # Internal containers ------------------------------------------------
        self._uuid_to_traces: "OrderedDict[str, object]" = OrderedDict()
        self._uuid_to_avg_adv: Dict[str, float] = {}
        self._uuid_order: Dict[str, int] = {}  # insertion step idx
        self._step_counter: int = 0            # global push counter
    # ------------------------------------------------------------------
    # Info helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._uuid_to_traces)

    def is_empty(self) -> bool:
        return len(self) == 0

    # ------------------------------------------------------------------
    # Public API – PUSH / SAMPLE
    # ------------------------------------------------------------------
    def push(self, batch) -> None:
        """Inspect *batch*, insert high‑adv uuids, then purge/evict as needed."""
        self._step_counter += 1  # == current *step* index (1‑based)

        # ------------------------------------------------------
        # Purge by age BEFORE we insert new uuids (keeps window tight)
        # ------------------------------------------------------
        if self.max_age_steps and self.max_age_steps > 0:
            threshold = self._step_counter - self.max_age_steps
            to_purge = [u for u, step in self._uuid_order.items() if step <= threshold]
            for u in to_purge:
                self._remove_uuid(u)
            if self.debug and to_purge:
                print(f"[ReplayBuffer] purge(age): removed {len(to_purge)} uuid(s): {to_purge}")

        # ------------------------------------------------------
        # Normal insertion logic (same as before)
        # ------------------------------------------------------
        uuids: List[str] = batch.non_tensor_batch["uuid"]
        advs: torch.Tensor = batch.batch["advantages"]
        attn_mask: torch.Tensor = batch.batch["attention_mask"]

        adv_row_mean = self.adv_mean_fn(advs, attn_mask)  # (N,)

        uuid_to_vals: Dict[str, List[float]] = defaultdict(list)
        uuid_to_idx: Dict[str, List[int]] = defaultdict(list)
        for idx, (uuid, val) in enumerate(zip(uuids, adv_row_mean.tolist())):
            uuid_to_vals[uuid].append(val)
            uuid_to_idx[uuid].append(idx)
        uuid_avg_adv = {u: sum(vs) / len(vs) for u, vs in uuid_to_vals.items()}
        n_uuid = len(uuid_avg_adv)
        if n_uuid == 0:
            return
        k_top = max(1, math.ceil(self.top_ratio * n_uuid))
        top_uuids = {
            u for u, _ in sorted(uuid_avg_adv.items(), key=lambda kv: kv[1], reverse=True)[:k_top]
        }

        if self.debug:
            print(
                f"[ReplayBuffer] push(step={self._step_counter}): incoming={n_uuid} uuid, "
                f"inserting={len(top_uuids)}"
            )

        inserted: List[str] = []
        for uuid in top_uuids:
            sub_batch = batch[uuid_to_idx[uuid]]
            if uuid in self._uuid_to_traces:  # refresh ordering
                self._remove_uuid(uuid, silent=True)
            self._uuid_to_traces[uuid] = sub_batch
            self._uuid_to_avg_adv[uuid] = uuid_avg_adv[uuid]
            self._uuid_order[uuid] = self._step_counter
            inserted.append(uuid)

        if self.debug and inserted:
            print("  + inserted:")
            for u in inserted:
                print(f"    {u}: avg_adv={self._uuid_to_avg_adv[u]:.4f}")

        # ------------------------------------------------------
        # Size‑based eviction (balanced score)
        # ------------------------------------------------------
        self._evict_until_fits()

    # ------------------------------------------------------------------
    def sample_for_training(self, size_batch):
        if self.is_empty():
            return None
        n_uuid = len(self)
        k_replay = max(1, math.ceil(self.replay_ratio * size_batch))
        sorted_items = sorted(
            self._uuid_to_avg_adv.items(), key=lambda kv: kv[1], reverse=True
        )
        print(f"len(sorted_items): {len(sorted_items)}")
        slice_top = sorted_items[:k_replay]
        self.rng.shuffle(slice_top)
        replay_uuids = [u for u, _ in slice_top]

        if self.debug:
            print(f"[ReplayBuffer] sample: k={k_replay}/{n_uuid}")
            for u in replay_uuids:
                age = self._step_counter - self._uuid_order[u]
                print(
                    f"    {u}: avg_adv={self._uuid_to_avg_adv[u]:.4f}, "
                    f"age={age} step(s)"
                )

        sub_batches = [self._uuid_to_traces[u] for u in replay_uuids]
        return _concat_batches(sub_batches)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _remove_uuid(self, uuid: str, *, silent: bool = False):
        """Remove a uuid from all internal structures (if present)."""
        self._uuid_to_traces.pop(uuid, None)
        self._uuid_to_avg_adv.pop(uuid, None)
        self._uuid_order.pop(uuid, None)
        if self.debug and not silent:
            print(f"[ReplayBuffer] remove: {uuid}")

    def _evict_until_fits(self) -> None:
        while len(self) > self.max_size:
            # Normalise advantage & recency for score computation
            adv_values = list(self._uuid_to_avg_adv.values())
            order_values = list(self._uuid_order.values())
            min_adv, max_adv = min(adv_values), max(adv_values)
            min_ord, max_ord = min(order_values), max(order_values)
            denom_adv = max_adv - min_adv + 1e-8
            denom_ord = max_ord - min_ord + 1e-8

            scores: Dict[str, float] = {}
            for u in self._uuid_to_traces:
                adv_norm = (self._uuid_to_avg_adv[u] - min_adv) / denom_adv
                recency_norm = (self._uuid_order[u] - min_ord) / denom_ord
                scores[u] = (
                    self.balance_alpha * adv_norm + (1.0 - self.balance_alpha) * recency_norm
                )

            min_score = min(scores.values())
            candidates = [u for u, s in scores.items() if abs(s - min_score) < 1e-12]
            uuid_to_remove = min(candidates, key=lambda u: self._uuid_order[u])
            if self.debug:
                print(
                    f"[ReplayBuffer] evict(size): {uuid_to_remove} | score={min_score:.4f} "
                    f"(adv={self._uuid_to_avg_adv[uuid_to_remove]:.4f}, "
                    f"age={self._step_counter - self._uuid_order[uuid_to_remove]})"
                )
            self._remove_uuid(uuid_to_remove, silent=True)