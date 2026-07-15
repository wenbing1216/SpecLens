import math
from collections import Counter
from typing import Iterable, Hashable, Sequence


def semantic_entropy_from_counts(cluster_sizes: Sequence[int], log_base: float = 2.0) -> float:
    """
    Compute semantic entropy from the sample counts of each semantic cluster.

    Args:
        cluster_sizes:
            Number of programs in each cluster.
            For example, [2, 18] means there are 20 programs in total:
            cluster 1 contains 2 programs and cluster 2 contains 18.

        log_base:
            Base of the logarithm. The 0.469 example in the paper uses log2,
            so the default is 2.

    Returns:
        The semantic entropy value.
    """
    if not cluster_sizes:
        raise ValueError("cluster_sizes must not be empty")

    if any(size < 0 for size in cluster_sizes):
        raise ValueError("cluster sizes must not be negative")

    total = sum(cluster_sizes)

    if total == 0:
        raise ValueError("the sum of all cluster sizes must not be 0")

    entropy = 0.0

    for size in cluster_sizes:
        if size == 0:
            continue

        p = size / total
        entropy -= p * math.log(p, log_base)

    return entropy
