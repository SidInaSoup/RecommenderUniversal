import pandas as pd
from recommender_universal.models.advanced.matrix_factorization import (
    MatrixFactorization,
)


def test_matrix_factorization_fit_and_recommend():
    # Sample interaction data
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": ["A", "B", "A", "C", "B", "D"],
            "rating": [5, 4, 3, 2, 4, 5],
        }
    )

    # Initialize and fit the model
    model = MatrixFactorization(
        user_col="user_id", item_col="item_id", rating_col="rating", factors=4, epochs=5
    )
    model.fit(df)

    # Make recommendations
    recs = model.recommend(user_id=1, k=2)

    # Assertions
    assert isinstance(recs, list)
    assert len(recs) == 2
    assert all(isinstance(item, str) for item in recs)
