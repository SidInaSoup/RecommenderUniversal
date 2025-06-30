import pandas as pd
from typing import Any, Callable, Dict, List, Set, Union

from .metrics import hit_rate_at_k, average_precision_at_k, ndcg_at_k  # noqa: F401


def evaluate_batch(
    df: pd.DataFrame,
    model: Any,
    k: int,
    metric_fn: Callable[[List[Any], Set[Any], int], float],
    user_col: str,
    item_col: str,
) -> float:
    """Average metric_fn over all users in df."""
    users = df[user_col].unique()
    scores: List[float] = []
    for user in users:
        relevant = set(df[df[user_col] == user][item_col])
        recs = model.recommend(user, k)
        scores.append(metric_fn(recs, relevant, k))
    return float(sum(scores) / len(scores)) if scores else 0.0


def stratified_evaluation(
    df: pd.DataFrame,
    model: Any,
    k: int,
    metric_fn: Callable[[List[Any], Set[Any], int], float],
    user_col: str,
    item_col: str,
    group_col: Union[str, pd.Grouper],
) -> Dict[Any, float]:
    """
    Compute the chosen metric_fn for each subgroup of the DataFrame.

    :param df: full test interactions
    :param model: recommender with .recommend(user, k)
    :param k: cutoff for recommendations
    :param metric_fn: one of hit_rate_at_k, average_precision_at_k, ndcg_at_k, etc
    :param user_col: column name for users
    :param item_col: column name for items
    :param group_col: either
        - a column name in df to group by, or
        - a `pd.Grouper` (e.g. pd.Grouper(key="timestamp", freq="M")) for time periods
    :return: dict mapping each group value → its batch metric score
    """
    results: Dict[Any, float] = {}
    # Use pandas grouping
    grouped = (
        df.groupby(group_col) if isinstance(group_col, str) else df.groupby(group_col)
    )
    for group_value, group_df in grouped:
        score = evaluate_batch(group_df, model, k, metric_fn, user_col, item_col)
        results[group_value] = score
    return results
