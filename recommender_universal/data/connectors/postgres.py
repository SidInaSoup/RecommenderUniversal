import pandas as pd
from sqlalchemy import create_engine
from .base import BaseConnector


class PostgresConnector(BaseConnector):
    def load(self, uri: str, query: str, **kwargs) -> pd.DataFrame:
        engine = create_engine(uri)
        df = pd.read_sql(query, engine, **kwargs)
        self._df = df
        return df


BaseConnector.register_connector("postgresql://", PostgresConnector)
BaseConnector.register_connector("postgres://", PostgresConnector)
