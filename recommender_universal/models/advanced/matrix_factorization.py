import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from recommender_universal.models.base import BaseRecommender
from recommender_universal.models.registry import register


@register("mf")
class MatrixFactorization(BaseRecommender):
    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
        factors: int = 10,
        lr: float = 0.01,
        epochs: int = 10,
    ) -> None:
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.factors = factors
        self.lr = lr
        self.epochs = epochs

        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[str, int] = {}
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> "MatrixFactorization":
        users = df[self.user_col].unique()
        items = df[self.item_col].unique()

        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}

        num_users = len(users)
        num_items = len(items)

        self.user_factors = np.random.normal(0, 0.1, (num_users, self.factors))
        self.item_factors = np.random.normal(0, 0.1, (num_items, self.factors))

        for _ in range(self.epochs):
            for _, row in df.iterrows():
                u_id = self.user_map[row[self.user_col]]
                i_id = self.item_map[row[self.item_col]]
                rating = row[self.rating_col]

                assert self.user_factors is not None and self.item_factors is not None
                pred = np.dot(self.user_factors[u_id], self.item_factors[i_id])
                err = rating - pred

                # SGD update
                self.user_factors[u_id] += self.lr * err * self.item_factors[i_id]
                self.item_factors[i_id] += self.lr * err * self.user_factors[u_id]

        return self

    def recommend(self, user_id: int, k: int = 5) -> List[str]:
        if user_id not in self.user_map:
            return []

        u_idx = self.user_map[user_id]
        assert self.user_factors is not None and self.item_factors is not None

        scores = np.dot(self.user_factors[u_idx], self.item_factors.T)
        top_indices = np.argsort(scores)[::-1][:k]

        reverse_item_map = {v: k for k, v in self.item_map.items()}
        return [reverse_item_map[i] for i in top_indices]
