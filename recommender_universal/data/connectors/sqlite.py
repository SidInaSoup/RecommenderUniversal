import sqlite3
import pandas as pd
from .base import BaseConnector


class SQLiteConnector(BaseConnector):
    def __init__(self, db_path: str, query: str):
        super().__init__()
        self.db_path = db_path
        self.query = query

    def _read(self) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(self.query, conn)
