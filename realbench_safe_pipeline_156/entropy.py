import math
from collections import Counter
from typing import Iterable, Hashable, Sequence


def semantic_entropy_from_counts(cluster_sizes: Sequence[int], log_base: float = 2.0) -> float:
    """
    根据每个 semantic cluster 的样本数量计算 semantic entropy。

    参数:
        cluster_sizes:
            每个 cluster 里有多少个程序。
            例如 [2, 18] 表示一共 20 个程序，cluster1 有 2 个，cluster2 有 18 个。

        log_base:
            对数底数。论文例子里的 0.469 对应 log2，所以默认用 2。

    返回:
        semantic entropy 数值。
    """
    if not cluster_sizes:
        raise ValueError("cluster_sizes 不能为空")

    if any(size < 0 for size in cluster_sizes):
        raise ValueError("cluster size 不能是负数")

    total = sum(cluster_sizes)

    if total == 0:
        raise ValueError("所有 cluster size 之和不能为 0")

    entropy = 0.0

    for size in cluster_sizes:
        if size == 0:
            continue

        p = size / total
        entropy -= p * math.log(p, log_base)

    return entropy