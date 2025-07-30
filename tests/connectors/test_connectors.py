import json  # noqa: F401
import sqlite3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fastavro
import s3fs  # noqa: F401
import pytest

from pathlib import Path  # noqa: F401

from recommender_universal.data.connectors.base import BaseConnector
from recommender_universal.data.connectors.csv import CSVConnector
from recommender_universal.data.connectors.json import JSONConnector
from recommender_universal.data.connectors.parquet import ParquetConnector
from recommender_universal.data.connectors.avro import AvroConnector
from recommender_universal.data.connectors.sqlite import SQLiteConnector

# Helper: uniform load_data convenience
from recommender_universal.data.connectors import load_data


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "item_id": ["A", "B", "C"],
            "rating": [5, 4, 3],
        }
    )


def test_csv_connector_load_and_cache(tmp_path, sample_df):
    path = tmp_path / "data.csv"
    sample_df.to_csv(path, index=False)

    # Registry lookup
    cls = BaseConnector.get_connector_for(str(path))
    assert cls is CSVConnector

    conn = cls(cache=True)
    # First load
    df1 = conn.load(str(path))
    pd.testing.assert_frame_equal(df1, sample_df)
    # Schema
    schema = conn.schema
    assert schema["user_id"].startswith("int")
    # Caching: modify file and reload without refresh → still old data
    modified = sample_df.copy()
    modified.loc[0, "user_id"] = 999
    modified.to_csv(path, index=False)
    df2 = conn.load(str(path), refresh=False)
    pd.testing.assert_frame_equal(df2, sample_df)
    # With refresh=True, sees new content
    df3 = conn.load(str(path), refresh=True)
    assert df3.loc[0, "user_id"] == 999


def test_csv_load_data_shortcut(tmp_path, sample_df):
    path = tmp_path / "data.csv"
    sample_df.to_csv(path, index=False)
    df = load_data(str(path))
    pd.testing.assert_frame_equal(df, sample_df)


def test_json_connector(tmp_path, sample_df):
    path = tmp_path / "data.json"
    sample_df.to_json(path, orient="records")

    cls = BaseConnector.get_connector_for(str(path))
    assert cls is JSONConnector

    conn = cls()
    df = conn.load(str(path), orient="records")
    pd.testing.assert_frame_equal(df, sample_df)


def test_parquet_connector(tmp_path, sample_df):
    path = tmp_path / "data.parquet"
    table = pa.Table.from_pandas(sample_df)
    pq.write_table(table, path)

    cls = BaseConnector.get_connector_for(str(path))
    assert cls is ParquetConnector

    conn = cls()
    df = conn.load(str(path))
    pd.testing.assert_frame_equal(df, sample_df)


def test_avro_connector(tmp_path, sample_df):
    path = tmp_path / "data.avro"
    records = sample_df.to_dict(orient="records")
    schema = {
        "doc": "Test",
        "name": "Test",
        "type": "record",
        "fields": [
            {"name": "user_id", "type": "int"},
            {"name": "item_id", "type": "string"},
            {"name": "rating", "type": "int"},
        ],
    }
    with open(path, "wb") as fo:
        fastavro.writer(fo, schema, records)

    cls = BaseConnector.get_connector_for(str(path))
    assert cls is AvroConnector

    conn = cls()
    df = conn.load(str(path))
    pd.testing.assert_frame_equal(df, sample_df.astype({"rating": int}))


def test_sqlite_connector(tmp_path, sample_df):
    db_path = tmp_path / "test.db"
    conn_sql = sqlite3.connect(db_path)
    sample_df.to_sql("ratings", conn_sql, index=False)
    conn_sql.close()

    uri = f"sqlite:///{db_path}"
    cls = BaseConnector.get_connector_for(uri)
    assert cls is SQLiteConnector

    conn = cls()
    df = conn.load(uri, table_name="ratings")
    pd.testing.assert_frame_equal(df, sample_df)

    # Via query
    df2 = conn.load(uri, query="SELECT * FROM ratings")
    df2 = df2.sort_values("user_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(df2, sample_df)


def test_get_connector_for_invalid():
    with pytest.raises(ValueError):
        BaseConnector.get_connector_for("unsupported://file.xyz")
