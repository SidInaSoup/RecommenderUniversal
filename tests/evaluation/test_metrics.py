from recommender_universal.evaluation.metrics import precision_at_k, recall_at_k


def test_precision_recall_at_k():
    recommended = [101, 102, 103, 104]
    relevant = {102, 104, 105}

    precision = precision_at_k(recommended, relevant, 3)
    recall = recall_at_k(recommended, relevant, 3)

    assert abs(precision - 1 / 3) < 1e-5
    assert abs(recall - 1 / 3) < 1e-5
