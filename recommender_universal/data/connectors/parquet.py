import pandas as pd
from .base import BaseConnector


class ParquetConnector(BaseConnector):
    def load(self, uri: str, **kwargs) -> pd.DataFrame:
        df = pd.read_parquet(uri, **kwargs)
        self._df = df
        return df


BaseConnector.register_connector(".parquet", ParquetConnector)
