from abc import ABC, abstractmethod
import pandas as pd


class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseRecommender":
        pass

    @abstractmethod
    def recommend(self, user_id: int, k: int = 5) -> list[int]:
        """
        Recommend k items for a given user.

        :param user_id: ID of the user to recommend items for.
        :param k: Number of items to recommend.
        :return: List of recommended item IDs.
        """
        pass

    def evaluate(self, test_df: pd.DataFrame) -> float:
        raise NotImplementedError("Evaluation not implemented")
