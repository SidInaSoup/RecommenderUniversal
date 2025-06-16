import pandas as pd
import os  # noqa: F401
from recommender_universal.models.advanced.matrix_factorization import (
    MatrixFactorization,
)


def test_matrix_factorization_save_load(tmp_path):
    # Create dummy data
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3],
            "item_id": ["A", "B", "A", "C", "B", "D"],
            "rating": [5, 4, 3, 2, 4, 5],
        }
    )

    # Initialize and fit the model
    model = MatrixFactorization(
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        factors=4,
        epochs=10,
    )
    model.fit(df)

    # Save to temporary path
    save_path = tmp_path / "mf_model.pkl"
    model.save(str(save_path))

    # Load into a new instance
    new_model = MatrixFactorization(
        user_col="user_id",
        item_col="item_id",
        rating_col="rating",
        factors=4,
        epochs=10,
    )
    new_model.load(str(save_path))

    # Check if recommendations work and are consistent
    old_recs = model.recommend(user_id=1, k=2)
    new_recs = new_model.recommend(user_id=1, k=2)

    assert len(old_recs) == len(new_recs)
    assert old_recs == new_recs
