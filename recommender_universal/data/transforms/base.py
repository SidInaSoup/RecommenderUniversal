from abc import (
    ABC,
    abstractmethod,
)  # ABC cannot be inherited unless all subclasses implement all abstract methods
import pandas as pd
from typing import TypeVar, Generic

T = TypeVar(
    "T", bound="BaseTransform"
)  # Generic type variable that must be a subclass of BaseTransform


class BaseTransform(ABC, Generic[T]):
    """
    Abstract base class for all transforms.
    """

    @abstractmethod  # Must be implemented by subclasses
    def fit(self, df: pd.DataFrame) -> T:
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
