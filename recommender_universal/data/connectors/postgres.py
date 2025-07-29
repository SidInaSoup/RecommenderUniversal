import pandas as pd
from sqlalchemy import create_engine
from .base import BaseConnector
from typing import Any


class PostgresConnector(BaseConnector):
    def _read(self, uri: str, query: str, **kwargs: Any) -> pd.DataFrame:
        engine = create_engine(uri)
        df = pd.read_sql(query, engine, **kwargs)
        self._df = df
        return df


BaseConnector.register_connector("postgresql://", PostgresConnector)
BaseConnector.register_connector("postgres://", PostgresConnector)
