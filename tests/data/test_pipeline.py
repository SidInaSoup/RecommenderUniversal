import pandas as pd
import pytest
from pathlib import Path  # noqa: F401

from recommender_universal.data.connectors.csv import CSVConnector
from recommender_universal.data.schema import RatingSchema
from recommender_universal.data.transforms.numerical import MinMaxScaler
from recommender_universal.data.pipeline import DataPipeline


def test_pipeline_minmax(tmp_path):
    # 1) Write out a tiny CSV
    path = tmp_path / "ratings.csv"
    df_in = pd.DataFrame(
        {
            "u": [1, 2, 3],
            "i": [10, 20, 10],
            "r": [3.0, 5.0, 4.0],
        }
    )
    df_in.to_csv(path, index=False)

    # 2) Build the pipeline
    pipeline = DataPipeline(
        connector=CSVConnector(cache=True),  # no URI in ctor anymore
        schema=RatingSchema(user="u", item="i", rating="r"),
        uri=str(path),  # pass the file path here
        transforms=[MinMaxScaler(columns=["r"])],
    )

    # 3) Run (validate + fit by default)
    df_out = pipeline.run()

    # 4) Assert shape and that 'r' has been scaled to [0,1]
    assert df_out.shape == df_in.shape
    assert df_out["r"].max() == pytest.approx(1.0)
    assert df_out["r"].min() == pytest.approx(0.0)
