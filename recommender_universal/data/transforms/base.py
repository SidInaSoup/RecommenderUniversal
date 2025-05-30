from abc import ABC, abstractmethod
import pandas as pd


class BaseTransform(ABC):
    """
    Abstract base class for all transforms.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the transform to the data.

        :param df: DataFrame to fit the transform on.
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
