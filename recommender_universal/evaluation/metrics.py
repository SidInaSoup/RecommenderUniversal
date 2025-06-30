import pandas as pd
from typing import List, Set, Callable, Protocol  # noqa: F401
import math


# class RecommenderProtocol(Protocol):
#     def recommend(self, user_id: int, k: int) -> list[int]: ...


# def hit_rate_at_k(
#     model: RecommenderProtocol,
#     test_df: pd.DataFrame,
#     user_column: str,
#     item_column: str,
#     k: int = 5,
# ) -> float:
#     """
#     Calculate Hit Rate at k for a recommender model.
#     """
#     hits = 0
#     total = 0

#     for user, group in test_df.groupby(user_column):
#         if group.empty:
#             continue
#         true_items = set(group[item_column])
#         preds = model.recommend(user, k)
#         hits += int(bool(true_items.intersection(preds)))
#         total += 1

# return hits / total if total else 0.0


# ----- Raw metrics -----


def hit_rate_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Raw Hit Rate@k: fraction of users where at least one relevant item is in top-k.
    """
    return float(any(item in relevant for item in recommended[:k]))


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    score = 0.0
    hits = 0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / math.log2(i + 2)

    ideal_dcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_batch(
    df: pd.DataFrame,
    model,
    k: int,
    metric_fn: Callable[[List[int], Set[int], int], float],
    user_col: str,
    item_col: str,
) -> float:
    """
    Applies a raw metric_fn for each user in df and returns the average score.
    """
    users = df[user_col].unique()
    scores = []
    for user in users:
        user_items = set(df[df[user_col] == user][item_col])
        recs = model.recommend(user, k)
        scores.append(metric_fn(recs, user_items, k))
    return sum(scores) / len(scores) if scores else 0.0
