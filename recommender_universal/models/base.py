import pandas as pd
import pickle
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List
from pathlib import Path  # noqa: F401

T = TypeVar("T")


class BaseRecommender(ABC, Generic[T]):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseRecommender":
        pass

    @abstractmethod
    def recommend(self, user_id: int, k: int = 5) -> List[T]:
        """
        Recommend k items for a given user.

        :param user_id: ID of the user to recommend items for.
        :param k: Number of items to recommend.
        :return: List of recommended item IDs.
        """
        pass

    def evaluate(self, test_df: pd.DataFrame) -> float:
        raise NotImplementedError("Evaluation not implemented")

    def save(self, path: str) -> None:
        """
        Save the model instance to a file using pickle.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str) -> None:
        """
        Load the model instance from a pickle file.
        Also loads dict dumps of model
        """
        with open(path, "rb") as f:
            loaded = pickle.load(f)

        if isinstance(loaded, dict):
            self.__dict__.update(loaded)

        elif hasattr(loaded, "__dict__"):
            self.__dict__.update(loaded.__dict__)
        else:
            raise ValueError(f"Unexpected data type in load(): {type(loaded)}")
