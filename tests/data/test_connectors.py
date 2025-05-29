import pandas as pd
import sqlite3
from recommender_universal.data.connectors.csv import CSVConnector
from recommender_universal.data.connectors.sqlite import SQLiteConnector


def test_csv_connector(tmp_path):
    file = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(file, index=False)

    df = CSVConnector(file).load()
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]


def test_sqlite_connector(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE users (id INTEGER, name TEXT);")
    conn.execute("INSERT INTO users VALUES (1, 'Alice'), (2, 'Bob');")
    conn.commit()
    conn.close()

    connector = SQLiteConnector(db_path=str(db_path), query="SELECT * FROM users")
    df = connector.load()

    assert df.shape == (2, 2)
    assert list(df.columns) == ["id", "name"]
