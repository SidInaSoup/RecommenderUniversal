import pandas as pd
import os  # noqa: F401
from recommender_universal.models.baseline.top_popular import TopPopularRecommender


def test_save_and_load(tmp_path):
    # Create dummy data
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "item_id": [101, 102, 101, 103],
            "rating": [5, 3, 4, 2],
        }
    )

    model = TopPopularRecommender(item_column="item_id")
    model.fit(df)

    # Save model
    save_path = tmp_path / "top_popular.pkl"
    model.save(str(save_path))

    # Create a new instance and load
    new_model = TopPopularRecommender(item_column="item_id")
    new_model.load(str(save_path))

    # Check if recommendation is consistent
    recs = new_model.recommend(user_id=1, k=2)
    assert len(recs) == 2
    assert all(isinstance(i, int) for i in recs)
