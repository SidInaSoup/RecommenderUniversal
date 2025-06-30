import pandas as pd
import math
import pytest

from recommender_universal.evaluation.metrics import (
    hit_rate_at_k,
    average_precision_at_k,
    ndcg_at_k,
)
from recommender_universal.evaluation.batch_eval import (
    evaluate_batch,
)

# Raw metric tests


def test_hit_rate_at_k():
    recommended = [1, 2, 3]
    relevant = {3, 4}
    assert hit_rate_at_k(recommended, relevant, 3) == 1.0
    assert hit_rate_at_k(recommended, relevant, 2) == 0.0


def test_average_precision_at_k():
    recommended = [1, 2, 3, 4]
    relevant = {2, 4}
    # hits at pos1 and pos3:
    # score = (1/2) + (2/4) = 0.5 + 0.5 = 1.0; denom = 2 -> AP = 0.5
    assert (
        pytest.approx(average_precision_at_k(recommended, relevant, 4), rel=1e-6) == 0.5
    )
    assert average_precision_at_k([1, 2, 3], set(), 3) == 0.0


def test_ndcg_at_k():

    recommended = [1, 2, 3, 4]
    relevant = {2, 4}
    # Compute DCG for actual ranking:
    # position i=0 item=1 → not relevant
    # i=1 item=2 → 1/log2(1+2)
    # i=2 item=3 → not relevant
    # i=3 item=4 → 1/log2(3+2)
    dcg = 1 / math.log2(1 + 2) + 1 / math.log2(3 + 2)

    # Ideal DCG for two relevant items at positions 0 & 1:
    # i=0 → 1/log2(0+2) = 1
    # i=1 → 1/log2(1+2)
    idcg = 1 + 1 / math.log2(1 + 2)

    expected = dcg / idcg
    result = ndcg_at_k(recommended, relevant, k=4)

    assert pytest.approx(result, rel=1e-6) == expected

    # No relevant items → zero
    assert ndcg_at_k([1, 2], set(), k=2) == 0.0


# Batch evaluation tests


def test_evaluate_batch_hit_rate():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": [10, 20, 10, 30],
        }
    )

    class Dummy:
        def recommend(self, u, k):
            return [10, 30][:k]

    model = Dummy()
    hr = evaluate_batch(
        df, model, k=2, metric_fn=hit_rate_at_k, user_col="user_id", item_col="item_id"
    )
    assert hr == 1.0


def test_evaluate_batch_average_precision():
    df = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [1, 2],
        }
    )

    class Dummy:
        def recommend(self, u, k):
            return [2, 1][:k]

    model = Dummy()
    ap = evaluate_batch(
        df,
        model,
        k=2,
        metric_fn=average_precision_at_k,
        user_col="user_id",
        item_col="item_id",
    )
    # For user 1: rec=[2,1], relevant={1,2}
    # AP = (1/1 + 2/2)/2 = (1 + 1)/2 = 1.0
    assert pytest.approx(ap, rel=1e-6) == 1.0
