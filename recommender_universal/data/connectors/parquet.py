import pandas as pd
from .base import BaseConnector
from typing import Any


class ParquetConnector(BaseConnector):
    def _read(self, uri: str, **kwargs: Any) -> pd.DataFrame:
        df = pd.read_parquet(uri, **kwargs)
        self._df = df
        return df


BaseConnector.register_connector(".parquet", ParquetConnector)
