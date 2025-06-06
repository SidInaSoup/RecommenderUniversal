import pandas as pd
from recommender_universal.models.baseline.top_popular import TopPopularRecommender


def test_top_popular():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 1, 3, 2],
            "item_id": [10, 10, 20, 10, 30],
        }
    )

    model = TopPopularRecommender().fit(df)
    recs = model.recommend(user_id=1, k=2)

    assert recs == [
        10,
        20,
    ], "TopPopularRecommender did not return expected recommendations"
