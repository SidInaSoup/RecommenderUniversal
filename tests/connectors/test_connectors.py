import json  # noqa: F401
import sqlite3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import fastavro
import boto3  # noqa: F401
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


def test_csv_connector(tmp_path, sample_df):
    # Write CSV
    path = tmp_path / "data.csv"
    sample_df.to_csv(path, index=False)

    # Registry should pick CSVConnector
    cls = BaseConnector.get_connector_for(str(path))
    assert cls is CSVConnector

    # Load via CSVConnector
    df = cls().load(str(path))
    pd.testing.assert_frame_equal(df, sample_df)

    # load_data shortcut
    df2 = load_data(str(path))
    pd.testing.assert_frame_equal(df2, sample_df)

    # Schema introspection
    conn = cls()
    conn.load(str(path))
    schema = conn.schema
    assert schema["user_id"].startswith("int")
    assert schema["item_id"] == "object"


def test_json_connector(tmp_path, sample_df):
    # Write JSON
    path = tmp_path / "data.json"
    sample_df.to_json(path, orient="records", lines=False)

    cls = BaseConnector.get_connector_for(str(path))
    assert cls is JSONConnector

    df = cls().load(str(path), orient="records")
    pd.testing.assert_frame_equal(df, sample_df)

    # load_data with .json
    df2 = load_data(str(path))
    pd.testing.assert_frame_equal(df2, sample_df)


def test_parquet_connector(tmp_path, sample_df):
    # Write Parquet
    path = tmp_path / "data.parquet"
    table = pa.Table.from_pandas(sample_df)
    pq.write_table(table, path)

    cls = BaseConnector.get_connector_for(str(path))
    assert cls is ParquetConnector

    df = cls().load(str(path))
    pd.testing.assert_frame_equal(df, sample_df)

    df2 = load_data(str(path))
    pd.testing.assert_frame_equal(df2, sample_df)


def test_avro_connector(tmp_path, sample_df):
    # Write Avro
    path = tmp_path / "data.avro"
    records = sample_df.to_dict(orient="records")
    schema = {
        "doc": "Test schema",
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

    df = cls().load(str(path))
    # Avro will read ints as ints, strings as objects
    pd.testing.assert_frame_equal(df, sample_df.astype({"rating": int}))

    df2 = load_data(str(path))
    pd.testing.assert_frame_equal(df2, sample_df.astype({"rating": int}))


def test_sqlite_connector(tmp_path, sample_df):
    # Create SQLite DB
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    sample_df.to_sql("ratings", conn, index=False)
    conn.close()

    uri = f"sqlite:///{db_path}"
    cls = BaseConnector.get_connector_for(uri)
    assert cls is SQLiteConnector

    # Load by table name
    df = cls().load(uri, table_name="ratings")
    pd.testing.assert_frame_equal(df, sample_df)

    # load_data helper with query
    query = "SELECT * FROM ratings"
    df2 = load_data(uri, query=query)
    # Order may vary, sort by user_id
    df2 = df2.sort_values("user_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(df2, sample_df)


def test_get_connector_for_invalid():
    with pytest.raises(ValueError):
        BaseConnector.get_connector_for("unsupported://path/data.xyz")
