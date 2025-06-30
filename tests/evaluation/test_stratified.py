import pandas as pd
import pytest

from recommender_universal.evaluation.batch_eval import stratified_evaluation
from recommender_universal.evaluation.metrics import hit_rate_at_k


class DummyModel:
    def recommend(self, user, k):
        return [1, 2, 3][:k]


def test_stratified_by_column():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": [1, 2, 1, 3, 2, 3],
            "country": ["US", "US", "UK", "UK", "US", "IN"],
        }
    )

    results = stratified_evaluation(
        df=df,
        model=DummyModel(),
        k=2,
        metric_fn=hit_rate_at_k,
        user_col="user_id",
        item_col="item_id",
        group_col="country",
    )

    # Based on implementation:
    # US group: users 1 & 3’s US row  -> both hit -> 1.0
    # UK group: user 2              -> hit         -> 1.0
    # IN group: user 3’s IN row     -> miss        -> 0.0
    assert results == {"IN": 0.0, "UK": 1.0, "US": 1.0}


def test_stratified_by_time_period():
    dates = pd.date_range("2025-01-01", periods=6, freq="15D")
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": [1, 2, 1, 3, 2, 3],
            "timestamp": dates,
        }
    ).sort_values("timestamp")

    # Group by month end
    results = stratified_evaluation(
        df=df,
        model=DummyModel(),
        k=1,
        metric_fn=hit_rate_at_k,
        user_col="user_id",
        item_col="item_id",
        group_col=pd.Grouper(key="timestamp", freq="ME"),
    )

    # As per grouping logic:
    # Jan 2025 (3 rows: users 1 & 2) -> both hit -> 1.0
    # Feb 2025 (1 row: user 2)       -> miss   -> 0.0
    # Mar 2025 (2 rows: user 3 twice) -> miss   -> 0.0
    months = sorted(results.keys())
    assert pytest.approx(results[months[0]], rel=1e-6) == 1.0
    assert pytest.approx(results[months[1]], rel=1e-6) == 0.0
    assert pytest.approx(results[months[2]], rel=1e-6) == 0.0
