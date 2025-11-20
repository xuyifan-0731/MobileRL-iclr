from __future__ import annotations
import os
import json
import math
import torch
import random
import numbers
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, List



class CurriculumManager(ABC):
    """
    Base class that maintains a mapping
    `task-index  ->  List[float] (execution scores)`.
    """

    # ---------- lifecycle --------------------------------------------------
    def __init__(self) -> None:
        # 每个索引对应一次或多次得分（float）；defaultdict 省去显式初始化
        self._records: DefaultDict[int, List[float]] = defaultdict(list)

    # ---------- abstract API ----------------------------------------------
    @abstractmethod
    def index_to_task(self, index: int) -> Any:  # noqa: D401
        """将整数索引转换为具体任务对象（由子类决定任务的表示形式）。"""
        raise NotImplementedError

    @abstractmethod
    def task_to_index(self, task: Any) -> int:  # noqa: D401
        """将任务对象映射回其整数索引。"""
        raise NotImplementedError

    @abstractmethod
    def sample_tasks(self, k: int) -> List[int]:  # noqa: D401
        """
        采样 k 个任务索引；采样策略由子类实现。
        返回值必须是合法索引 (int) 的列表。
        """
        raise NotImplementedError

    # ---------- public helpers --------------------------------------------
    def update(self,  task_array: List[str] = None, reward_list: List[float] = None, new_scores: Dict[int, List[float]] = None) -> None:
        """
        批量更新成绩记录。
        参数格式:
            { index (int): [score1 (float), score2 (float), ...], ... }
        所有合法条目会被追加到各自的历史记录中。
        """

        print("========in update,task_array",task_array)
        print("========in update,reward_list",reward_list)
        print("========in update,new_scores",new_scores)
        # new_scores和task_array不能同时为none
        # assert new_scores is not None or (task_array is not None and reward_list is not None), "new_scores和task_array不能同时为none"

        if new_scores is None:
            new_scores = defaultdict(list)
            for index, reward in zip(task_array, reward_list):
                new_scores[index].append(reward)
            
        print("========in update,new_scores",new_scores)
        if not isinstance(new_scores, dict):
            raise TypeError("update() 需要 dict[int, list[float]] 作为参数")

        for idx, scores in new_scores.items():
            # ---- key 校验 ---------------------------------------------------
            _, idx = self.index_to_task(idx)
            if not isinstance(idx, int):
                raise TypeError(f"索引 {idx!r} 不是 int")
            if idx < 0:
                raise ValueError(f"索引必须为非负整数，收到 {idx}")

            # ---- value 校验 -------------------------------------------------
            if not isinstance(scores, list):
                raise TypeError(f"索引 {idx} 的成绩应为 list[float]，收到 {type(scores)}")

            for s in scores:
                if not isinstance(s, numbers.Real):
                    print(s)
                    raise TypeError(
                        f"索引 {idx} 的成绩 {s!r} 不是数字 (int / float)"
                    )
                self._records[idx].append(float(s))
    
    def from_json(self, path: str) -> None:
        """
        读取 JSON 并把分数写入 *当前对象* 的记录表。

        JSON 格式示例
        ------------
        {
          "MarkorMoveNote":            [1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
          "ExpenseDeleteDuplicates":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          "SimpleCalendarAnyEventsOnDate": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          …
        }

        Notes
        -----
        * 仅修改 `self._records`；不会影响其他状态。
        * 若 JSON 含未知任务名，抛 `ValueError`。
        """
        import json, os, numbers

        if not isinstance(path, str):
            raise TypeError("path 必须是 str")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"文件不存在: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise TypeError("JSON 顶层应为对象 (dict)")

        # 将 task 名 -> 内部索引
        translated: Dict[int, List[float]] = {}
        for task_name, scores in raw.items():
            if task_name not in self._task_to_idx:
                raise ValueError(f"JSON 中未知任务 {task_name!r}")
            if not isinstance(scores, list):
                raise TypeError(f"{task_name} 对应值应为 list[float]")
            for s in scores:
                if not isinstance(s, numbers.Real):
                    raise TypeError(f"{task_name} 里的分数 {s!r} 不是数字")
            translated[self._task_to_idx[task_name]] = scores

        # 复用 update() 做最终写入与深层次校验
        self.update(new_scores=translated)

    # ---------- convenient accessors --------------------------------------
    def task_history(self, index: int) -> List[float]:
        """返回指定任务索引的完整历史得分（拷贝）。若无记录则返回空列表。"""
        if not isinstance(index, int):
            raise TypeError("index 必须为 int")
        return list(self._records.get(index, []))

    def __len__(self) -> int:
        """当前已记录的任务数量（不同索引的个数）。"""
        return len(self._records)

    # ---------- debug & display -------------------------------------------
    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"<{cls} tasks={len(self)}>"

class GaussianCurriculumManager(CurriculumManager):
    _SIGMA = math.sqrt(0.25 / (2 * math.log(5)))  # 0/100 ≈ 1/5 权重
    _HIST_FACTOR = 3                               # 避免最近 k×3 次重复

    def __init__(self) -> None:
        from collections import defaultdict
        self._records: Dict[int, List[float]] = defaultdict(list)
        self._sample_hist: deque[int] = deque(maxlen=10_000)

    # 抽象方法占位（这里直接抛错，子类会覆盖） ------------------------------
    def index_to_task(self, index: int): raise NotImplementedError
    def task_to_index(self, task: Any): raise NotImplementedError

    # ----------------- 高斯采样 ------------------------------------------
    def _weighted_sample(self, pool: List[int], k: int,
                         weights: Dict[int, float]) -> List[int]:
        sel, cand = [], pool.copy()
        while len(sel) < k:
            w_sum = sum(weights[i] for i in cand)
            probs = [weights[i] / w_sum for i in cand]
            pick = random.choices(cand, probs, k=1)[0]
            sel.append(pick)
            cand.remove(pick)
        return sel

    def _gaussian_weights(self) -> Dict[int, float]:
        """给每个内部任务索引计算采样权重。
        · μ == 0           → 权重 0（彻底排除）
        · 0.2 ≤ μ ≤ 0.8    → 权重 1
        · 其余            → 高斯衰减（0/1 处 ≈ 0.2）
        """
        w: Dict[int, float] = {}
        for idx, scores in self._records.items():
            mu = sum(scores) / len(scores) if scores else 0.5

            #if mu == 0:                        # ← 新增：全 0 的任务不采样
                #w[idx] = 0.0
            if 0.2 <= mu <= 0.8:
                w[idx] = 1.0
            else:
                w[idx] = math.exp(-((mu - 0.5) ** 2) / (2 * self._SIGMA ** 2))
        return w
# ──────────────────────────────────────────────────────────────────────────



class AndroidWorldCurriculumManager(CurriculumManager):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.config = config
        index_cfg = config.data.gen_chat.train
        task_list_path = config.data.curriculum.task
        
        # 加载任务列表
        if os.path.exists(task_list_path):
            with open(task_list_path, "r") as f:
                task_list = json.load(f)
        else:
            raise ValueError(f"task_list not found: {task_list_path}")
        
        # 加载invalid任务
        invalid_tasks = config.data.curriculum.get("invalid_tasks", [])
        if invalid_tasks != []:
            if os.path.exists(invalid_tasks):   
                with open(invalid_tasks, "r") as f:
                    invalid_tasks = json.load(f)
            else:
                raise ValueError(f"invalid_tasks not found: {invalid_tasks}")

        # 解析并验证 index_cfg
        for key in ("name", "index_start", "index_end"):
            if key not in index_cfg:
                raise ValueError(f"index_cfg 缺少 {key!r}")
        self.name: str = str(index_cfg["name"])
        self._idx_start: int = int(index_cfg["index_start"])
        self._idx_end: int = int(index_cfg["index_end"])
        if self._idx_start < 0 or self._idx_end < self._idx_start:
            raise ValueError("index_start / index_end 范围非法")

        # 保存任务列表并初始化记录
        if not task_list:
            raise ValueError("task_list 不能为空")
        self._task_list: List[str] = list(task_list)
        self._task_to_idx: Dict[str, int] = {
            t: i for i, t in enumerate(self._task_list)
        }
        for i in range(len(self._task_list)):
            self._records[i]  # 创建空 list
        
        # 初始化失败计数：记录每个任务的连续失败次数（reward为0的次数）
        self._failure_counts: Dict[int, int] = defaultdict(int)
        
        # 从初始化文件加载历史记录
        if hasattr(config.data.curriculum, 'init'):
            self.from_json(config.data.curriculum.init)

        # 处理 invalid_tasks
        invalid_tasks = invalid_tasks or []
        unknown = set(invalid_tasks) - set(self._task_to_idx)
        if unknown:
            raise ValueError(f"invalid_tasks 中存在未知任务: {unknown}")
        self._invalid_idx: set[int] = {
            self._task_to_idx[t] for t in invalid_tasks
        }

    def index_to_task(self, index: int) -> tuple[str, int]:
        """将外部索引转换为任务名称和内部索引"""
        if not isinstance(index, int):
            raise TypeError("index 必须是 int")
        n = len(self._task_list)
        if n == 0:
            raise RuntimeError("尚无任务可映射")
        internal_index = index % n
        return self._task_list[internal_index], internal_index

    def task_to_index(self, task: str) -> int:
        """将任务名称转换为外部索引"""
        if task not in self._task_to_idx:
            raise ValueError(f"未知任务 {task!r}")
        n = len(self._task_list)
        task_id = self._task_to_idx[task]

        # 找到区间内首个满足余数条件的索引
        offset = (task_id - (self._idx_start % n)) % n
        first = self._idx_start + offset
        if first > self._idx_end:
            raise RuntimeError("合法区间内不存在对应索引")

        step = n
        candidates = list(range(first, self._idx_end + 1, step))
        return random.choice(candidates)

    def update(self, task_array: List[str] = None, reward_list: List[float] = None, new_scores: Dict[int, List[float]] = None) -> None:
        """
        批量更新成绩记录，并更新失败计数。
        当reward为0时，增加失败计数；当reward不为0时，重置失败计数。
        注意：按照得分顺序处理，每个得分都会影响失败计数。
        """
        # 先更新失败计数（在调用父类update之前）
        if new_scores is None:
            new_scores = defaultdict(list)
            for index, reward in zip(task_array, reward_list):
                new_scores[index].append(reward)
        
        # 更新失败计数：按照得分顺序处理
        for idx, scores in new_scores.items():
            # 获取内部索引
            _, internal_idx = self.index_to_task(idx)
            
            # 按顺序处理每个得分
            for score in scores:
                if score == 0.0:
                    # reward为0，增加失败计数
                    self._failure_counts[internal_idx] += 1
                else:
                    # reward不为0，重置失败计数
                    self._failure_counts[internal_idx] = 0
        
        # 调用父类的update方法更新记录
        super().update(task_array=task_array, reward_list=reward_list, new_scores=new_scores)

    def calculate_accuracy(self, index: int) -> float:
        """计算指定任务的正确率"""
        scores = self._records.get(index, [])
        if not scores:
            return 0.0  # 如果没有记录，返回0
        return sum(scores) / len(scores)

    def _calculate_sampling_weights(self) -> Dict[int, float]:
        """
        根据失败次数计算每个任务的采样权重
        - 失败次数 f <= 3: 权重为 exp(-f)
        - 失败次数 f > 3: 权重为 0
        """
        weights: Dict[int, float] = {}
        for idx in self._records.keys():
            failure_count = self._failure_counts.get(idx, 0)
            if failure_count > 3:
                weights[idx] = 0.0
            else:
                weights[idx] = math.exp(-failure_count)
        return weights

    def _weighted_sample(self, pool: List[int], k: int, weights: Dict[int, float]) -> List[int]:
        """
        根据权重进行加权采样
        
        Args:
            pool: 候选任务索引列表
            k: 需要采样的数量
            weights: 每个任务的权重字典
            
        Returns:
            采样得到的任务索引列表
        """
        # 过滤掉权重为0的任务
        valid_pool = [i for i in pool if weights.get(i, 0.0) > 0.0]
        
        if len(valid_pool) < k:
            raise ValueError(
                f"权重非零的任务仅 {len(valid_pool)} 个，无法抽取 {k} 个"
            )
        
        # 计算权重列表
        weight_list = [weights.get(i, 0.0) for i in valid_pool]
        
        # 使用加权随机采样
        chosen_internal = []
        remaining_pool = valid_pool.copy()
        remaining_weights = weight_list.copy()
        
        for _ in range(k):
            if not remaining_pool:
                break
            
            # 归一化权重
            total_weight = sum(remaining_weights)
            if total_weight == 0:
                # 如果所有权重都为0，使用均匀采样
                chosen = random.choice(remaining_pool)
            else:
                probs = [w / total_weight for w in remaining_weights]
                chosen = random.choices(remaining_pool, weights=probs, k=1)[0]
            
            chosen_internal.append(chosen)
            
            # 从候选池中移除已选中的任务
            idx = remaining_pool.index(chosen)
            remaining_pool.pop(idx)
            remaining_weights.pop(idx)
        
        return chosen_internal

    def sample_tasks(self, k: int) -> tuple[List[int], List[float]]:
        """
        根据失败次数进行加权采样k个任务，返回任务索引和对应的正确率
        
        Args:
            k (int): 需要采样的任务数量
            
        Returns:
            tuple[List[int], List[float]]: (任务索引列表, 正确率列表)
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k 必须为正整数")

        # 1) 获取有效任务池（排除invalid任务）
        valid_idx = [i for i in self._records if i not in self._invalid_idx]
        if len(valid_idx) < k:
            raise ValueError(
                f"合法可采样任务仅 {len(valid_idx)} 个，无法抽取 {k} 个"
            )

        # 2) 计算采样权重
        weights = self._calculate_sampling_weights()
        
        # 3) 根据权重进行加权采样
        chosen_internal = self._weighted_sample(valid_idx, k, weights)

        # 4) 转换为外部索引
        external_indices: List[int] = []
        accuracies: List[float] = []
        
        for internal_id in chosen_internal:
            task_name = self._task_list[internal_id]
            ext_idx = self.task_to_index(task_name)
            
            # 确保不重复
            while ext_idx in external_indices:
                ext_idx = self.task_to_index(task_name)
            
            external_indices.append(ext_idx)
            accuracies.append(self.calculate_accuracy(internal_id))

        # 5) 打印采样信息（包括失败次数和权重）
        diag = [
            (
                self._task_list[i],
                round(self.calculate_accuracy(i), 3),
                self._failure_counts.get(i, 0),
                round(weights.get(i, 0.0), 3)
            )
            for i in chosen_internal
        ]
        print("Sampled tasks (name, accuracy, failure_count, weight):", diag)

        print("========in sample_tasks,external_indices",external_indices)
        

        return external_indices, accuracies

    def sample_batch(self, batch_dict: Dict[str, Any], n: int) -> tuple[Dict[str, Any], List[int]]:
        """
        采样任务索引并更新 batch_dict，根据准确率动态调整采样次数
        
        Args:
            batch_dict (dict): 包含任务名称的 batch 字典
            n (int): 每条数据原本的采样次数
            
        Returns:
            tuple[dict, List[int]]: (更新后的batch_dict, 每个任务的采样次数列表)
        """
        # 采样任务数量
        k = len(batch_dict["index"])
        sampled_indices, accuracies = self.sample_tasks(k)

        # 根据准确率计算采样次数
        repeat_counts = self._calculate_sample_counts(accuracies, n)
        
        # 更新batch_dict
        batch_dict["index"] = torch.tensor(sampled_indices)

        print("========in sample_batch, accuracies:", accuracies)
        print("========in sample_batch, repeat_counts:", repeat_counts)
        
        return batch_dict, repeat_counts

    def _calculate_sample_counts(self, accuracies: List[float], n: int) -> List[int]:
        """
        根据准确率计算每个任务的采样次数
        
        Args:
            accuracies (List[float]): 每个任务的准确率
            n (int): 基础采样次数
            
        Returns:
            List[int]: 每个任务的采样次数
        """
        k = len(accuracies)
        total_samples = k * n  # 总采样次数必须保持不变
        
        min_samples = max(1, n // 4)  # 最小采样次数，准确率为1的任务
        max_samples = n * 2  # 最大采样次数，准确率为0的任务
        
        # 确保边界合理
        if min_samples >= max_samples:
            # 如果范围太小，直接平均分配
            return [n] * k
        
        # 使用平滑的反比例函数计算权重
        # 采用 sigmoid-like 函数使得变化更平滑
        weights = []
        for acc in accuracies:
            # 将准确率映射到 [0, 1] 范围，然后使用平滑函数
            acc = max(0.0, min(1.0, acc))  # 确保在 [0, 1] 范围内
            
            # 使用平滑的反比例映射：
            # acc = 0 -> weight = max_samples
            # acc = 1 -> weight = min_samples  
            # 使用平滑的非线性函数获得更好的分布
            # weight = min_samples + (max_samples - min_samples) * (1 - acc)^2
            weight = min_samples + (max_samples - min_samples) * ((1 - acc) ** 1.5)
            weights.append(weight)
        
        # 标准化权重，使总和等于total_samples
        weight_sum = sum(weights)
        if weight_sum == 0:
            return [n] * k
        
        # 按比例缩放权重
        normalized_weights = [w * total_samples / weight_sum for w in weights]
        
        # 使用更智能的取整方法
        sample_counts = []
        cumulative_error = 0.0
        
        for weight in normalized_weights:
            # 累积取整误差，当误差达到 0.5 时进位
            expected = weight + cumulative_error
            rounded = round(expected)
            sample_counts.append(max(min_samples, min(max_samples, rounded)))
            cumulative_error = expected - rounded
        
        # 调整总和到目标值
        current_sum = sum(sample_counts)
        diff = total_samples - current_sum
        
        # 创建调整优先级：准确率低的任务优先获得更多采样次数
        priority_indices = sorted(range(k), key=lambda i: (accuracies[i], i))
        
        # 调整采样次数
        adjustment_idx = 0
        while diff != 0:
            idx = priority_indices[adjustment_idx % k]
            
            if diff > 0:
                # 需要增加采样次数
                if sample_counts[idx] < max_samples:
                    sample_counts[idx] += 1
                    diff -= 1
            else:
                # 需要减少采样次数，从准确率高的任务开始
                reverse_idx = priority_indices[-(adjustment_idx % k) - 1]
                if sample_counts[reverse_idx] > min_samples:
                    sample_counts[reverse_idx] -= 1
                    diff += 1
            
            adjustment_idx += 1
            
            # 防止无限循环
            if adjustment_idx > k * 2:
                break
        
        # 最终确保约束满足
        for i in range(k):
            sample_counts[i] = max(min_samples, min(max_samples, sample_counts[i]))
        
        # 验证总和
        final_sum = sum(sample_counts)
        if final_sum != total_samples:
            print(f"Warning: 采样次数总和不匹配，期望 {total_samples}，实际 {final_sum}")
            # 如果还是不匹配，随机调整
            diff = total_samples - final_sum
            import random
            indices = list(range(k))
            random.shuffle(indices)
            
            for idx in indices:
                if diff == 0:
                    break
                if diff > 0 and sample_counts[idx] < max_samples:
                    sample_counts[idx] += 1
                    diff -= 1
                elif diff < 0 and sample_counts[idx] > min_samples:
                    sample_counts[idx] -= 1
                    diff += 1
        
        return sample_counts

    def get_task_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有任务的统计信息
        
        Returns:
            dict: 任务名称 -> {accuracy: float, count: int, scores: List[float]}
        """
        stats = {}
        for internal_id, scores in self._records.items():
            if internal_id not in self._invalid_idx:
                task_name = self._task_list[internal_id]
                stats[task_name] = {
                    'accuracy': self.calculate_accuracy(internal_id),
                    'count': len(scores),
                    'scores': scores.copy()
                }
        return stats

    def get_valid_task_count(self) -> int:
        """获取有效任务数量（排除invalid任务）"""
        return len([i for i in self._records if i not in self._invalid_idx])

'''
class AndroidWorldCurriculumManager(GaussianCurriculumManager):
    """
    CurriculumManager 具体实现：
    · 支持 task_list 初始化及 invalid_tasks 排采
    · 继承高斯采样 + “最近 k×3” 去重策略
    """

    # ---------- initialisation -------------------------------------------
    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()

        self.config = config
        index_cfg = config.data.gen_chat.train
        task_list_path = config.data.curriculum.task
        if os.path.exists(task_list_path):
            with open(task_list_path, "r") as f:
                task_list = json.load(f)
        else:
            raise ValueError(f"task_list not found: {task_list_path}")
        invalid_tasks = config.data.curriculum.get("invalid_tasks", [])
        if invalid_tasks != []:
            if os.path.exists(invalid_tasks):   
                with open(invalid_tasks, "r") as f:
                    invalid_tasks = json.load(f)
            else:
                raise ValueError(f"invalid_tasks not found: {invalid_tasks}")

        # —— 1) 解析并验证 index_cfg ——————————
        for key in ("name", "index_start", "index_end"):
            if key not in index_cfg:
                raise ValueError(f"index_cfg 缺少 {key!r}")
        self.name: str = str(index_cfg["name"])
        self._idx_start: int = int(index_cfg["index_start"])
        self._idx_end: int = int(index_cfg["index_end"])
        if self._idx_start < 0 or self._idx_end < self._idx_start:
            raise ValueError("index_start / index_end 范围非法")

        # —— 2) 保存 task_list 并初始化记录 ————
        if not task_list:
            raise ValueError("task_list 不能为空")
        self._task_list: List[str] = list(task_list)
        self._task_to_idx: Dict[str, int] = {
            t: i for i, t in enumerate(self._task_list)
        }
        for i in range(len(self._task_list)):
            self._records[i]  # 创建空 list
        self.from_json(config.data.curriculum.init)

        # —— 3) 处理 invalid_tasks ————————————
        invalid_tasks = invalid_tasks or []
        unknown = set(invalid_tasks) - set(self._task_to_idx)
        if unknown:
            raise ValueError(f"invalid_tasks 中存在未知任务: {unknown}")
        self._invalid_idx: set[int] = {
            self._task_to_idx[t] for t in invalid_tasks
        }

    # ---------- index ↔ task ---------------------------------------------
    def index_to_task(self, index: int) -> str:
        if not isinstance(index, int):
            raise TypeError("index 必须是 int")
        n = len(self._task_list)
        if n == 0:
            raise RuntimeError("尚无任务可映射")
        return self._task_list[index % n], index % n

    def task_to_index(self, task: str) -> int:
        if task not in self._task_to_idx:
            raise ValueError(f"未知任务 {task!r}")
        n = len(self._task_list)
        task_id = self._task_to_idx[task]

        # 找到区间内首个满足余数条件的索引
        offset = (task_id - (self._idx_start % n)) % n
        first = self._idx_start + offset
        if first > self._idx_end:
            raise RuntimeError("合法区间内不存在对应索引")

        step = n
        candidates: Sequence[int] = range(first, self._idx_end + 1, step)
        return random.choice(list(candidates))

    # ---------- sampling --------------------------------------------------
    def sample_tasks(self, k: int) -> List[int]:
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k 必须为正整数")

        # 1) 有效任务池 = 全部内部索引 - invalid
        valid_idx = [i for i in self._records if i not in self._invalid_idx]
        if len(valid_idx) < k:
            raise ValueError(
                f"合法可采样任务仅 {len(valid_idx)} 个，无法抽取 {k} 个"
            )

        # 2) 计算权重（仅对 valid 部分采样）
        weights_all = self._gaussian_weights()                      # ← 0.2–0.8 等权，其余高斯
        weights = {i: w for i, w in weights_all.items() if i in valid_idx}

        # 3) 按“最近 k×3” 规避重复
        recent = set(list(self._sample_hist)[-self._HIST_FACTOR * k :])
        primary = [i for i in valid_idx if i not in recent]
        secondary = [i for i in valid_idx if i in recent]

        chosen_internal: List[int] = []
        need = k
        if primary:
            take = min(len(primary), need)
            chosen_internal += self._weighted_sample(primary, take, weights)
            need -= take
        if need:
            chosen_internal += self._weighted_sample(secondary, need, weights)

        # 4) 转换为“外部 index”并保证唯一
        print("========in curriculum,self._records",self._records)
        print("========in sample_tasks,chosen_internal",chosen_internal)
        external_indices: List[int] = []
        for internal_id in chosen_internal:
            task_name = self._task_list[internal_id]
            ext_idx = self.task_to_index(task_name)   # ← 随机合法索引
            # 理论上不同 task 的余数不同，不会撞；仍做保险检查
            while ext_idx in external_indices:
                ext_idx = self.task_to_index(task_name)
            external_indices.append(ext_idx)
        
        diag = [
            (
                self._task_list[i],
                round(sum(self._records[i]) / len(self._records[i]), 3)
                if self._records[i] else None
            )
            for i in chosen_internal
        ]
        print("Sampled tasks (name, acc):", diag)         # ← 这一行即可

        # 5) 记录历史（仍记录内部 id，便于后续“最近 k×3” 逻辑）
        self._sample_hist.extend(chosen_internal)
        return external_indices

    def sample_batch(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        采样任务索引并更新 batch_dict。
        
        Args:
            batch_dict (dict): 包含任务名称的 batch 字典。
        
        Returns:
            dict: 更新后的 batch_dict。
        """
        # 1) 采样任务数量（一个tensor）

        sampled_indices = self.sample_tasks(len(batch_dict["index"]))

        # 2) 更新batch_dict，转化为tensor
        batch_dict["index"] = torch.tensor(sampled_indices)
        return batch_dict
'''

def test_calculate_sample_counts():
    """测试 _calculate_sample_counts 方法"""
    # 模拟一个简单的配置
    class MockConfig:
        def __init__(self):
            self.data = MockData()
    
    class MockData:
        def __init__(self):
            self.gen_chat = MockGenChat()
            self.curriculum = MockCurriculum()
    
    class MockGenChat:
        def __init__(self):
            self.train = {"name": "test", "index_start": 0, "index_end": 100}
    
    class MockCurriculum:
        def __init__(self):
            self.task = "test_tasks.json"
        
        def get(self, key, default=None):
            return default
    
    # 创建模拟任务数据
    import json
    import os
    test_tasks = ["task1", "task2", "task3", "task4", "task5"]
    with open("test_tasks.json", "w") as f:
        json.dump(test_tasks, f)
    
    try:
        config = MockConfig()
        cm = AndroidWorldCurriculumManager(config)
        
        # 测试不同准确率的情况
        accuracies = [0.0, 0.25, 0.5, 0.75, 1.0]  # 5个任务，不同准确率
        n = 8  # 基础采样次数
        
        repeat_counts = cm._calculate_sample_counts(accuracies, n)
        
        print("=== 测试结果 ===")
        print(f"准确率: {accuracies}")
        print(f"基础采样次数: {n}")
        print(f"采样次数分配: {repeat_counts}")
        print(f"总采样次数: {sum(repeat_counts)} (期望: {len(accuracies) * n})")
        print(f"最小采样次数: {min(repeat_counts)} (期望最小: {max(1, n//4)})")
        print(f"最大采样次数: {max(repeat_counts)} (期望最大: {n*2})")
        
        # 验证约束
        assert sum(repeat_counts) == len(accuracies) * n, "总采样次数不匹配"
        assert min(repeat_counts) >= max(1, n//4), "最小采样次数违反约束"
        assert max(repeat_counts) <= n*2, "最大采样次数违反约束"
        
        print("✅ 所有约束检查通过！")
        
        # 验证准确率和采样次数的关系
        for i in range(len(accuracies)):
            for j in range(i+1, len(accuracies)):
                if accuracies[i] < accuracies[j]:
                    if repeat_counts[i] < repeat_counts[j]:
                        print(f"⚠️  准确率 {accuracies[i]} < {accuracies[j]}，但采样次数 {repeat_counts[i]} < {repeat_counts[j]}")
                    else:
                        print(f"✓ 准确率 {accuracies[i]} < {accuracies[j]}，采样次数 {repeat_counts[i]} >= {repeat_counts[j]}")
        
        print("✅ 准确率与采样次数关系检查通过！")
        
        # 额外测试：不同的n值
        for test_n in [2, 4, 12, 16]:
            test_counts = cm._calculate_sample_counts(accuracies, test_n)
            print(f"\nn={test_n}: {test_counts}, 总和={sum(test_counts)}, 期望={len(accuracies)*test_n}")
        
    finally:
        # 清理测试文件
        if os.path.exists("test_tasks.json"):
            os.remove("test_tasks.json")
