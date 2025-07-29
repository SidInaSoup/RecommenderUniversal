import pandas as pd
from sqlalchemy import create_engine
from .base import BaseConnector
from typing import Any, Optional


class SQLiteConnector(BaseConnector):
    def _read(
        self,
        uri: str,
        table_name: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs: Any,
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
