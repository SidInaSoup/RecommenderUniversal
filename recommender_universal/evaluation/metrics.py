import pandas as pd


def hit_rate_at_k(
    model,
    test_df: pd.DataFrame,
    user_column: str,
    item_column: str,
    k: int = 5,
) -> float:
    """
    Calculate Hit Rate at k for a recommender model.
    """
    hits = 0
    total = 0

    for user, group in test_df.groupby(user_column):
        if group.empty:
            continue
        true_items = set(group[item_column])
        preds = model.recommend(user, k)
        hits += int(bool(true_items.intersection(preds)))
        total += 1

    return hits / total if total else 0.0
