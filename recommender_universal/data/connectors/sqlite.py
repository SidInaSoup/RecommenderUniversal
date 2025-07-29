import pandas as pd
from sqlalchemy import create_engine
from .base import BaseConnector


class SQLiteConnector(BaseConnector):
    def load(
        self, uri: str, table_name: str = None, query: str = None, **kwargs
    ) -> pd.DataFrame:
        engine = create_engine(uri)
        if query:
            df = pd.read_sql(query, engine, **kwargs)
        else:
            df = pd.read_sql_table(table_name, engine, **kwargs)
        self._df = df
        return df


BaseConnector.register_connector("sqlite://", SQLiteConnector)
BaseConnector.register_connector(".db", SQLiteConnector)
