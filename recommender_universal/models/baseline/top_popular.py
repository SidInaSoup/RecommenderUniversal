import pandas as pd
from collections import Counter
from recommender_universal.models.base import BaseRecommender


class TopPopularRecommender(BaseRecommender):

    def __init__(self, item_column: str = "item_id") -> None:
        self.item_column = item_column
        self.top_items: list[int] = []

    def fit(self, df: pd.DataFrame) -> "TopPopularRecommender":
        """
        Fit TopPopularRecommender Model to dataframe

        :param df: user-item interaction matrix
        :return: TopPopularRecommender instance
        """

        counter = Counter(df[self.item_column])
        self.top_items = [item for item, _ in counter.most_common()]
        return self

    def recommend(self, user_id: int, k: int = 5) -> list[int]:
        return self.top_items[:k]
