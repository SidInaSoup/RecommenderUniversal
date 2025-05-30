import pandas as pd
from .base import BaseTransform


class MinMaxScaler(BaseTransform):
    """
    Min-Max Scaler for normalizing numerical features to a range [0, 1].
    """

    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.mins: dict[str, float] = {}
        self.maxs: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "MinMaxScaler":
        for col in self.columns:
            self.mins[col] = df[col].min()
            self.maxs[col] = df[col].max()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in self.columns:
            min_, max_ = self.mins[col], self.maxs[col]
            out[col] = (df[col] - min_) / (max_ - min_ + 1e-8)  # Avoid division by zero
        return out
