import pandas as pd
from recommender_universal.data.connectors.csv import CSVConnector
from recommender_universal.data.schema import RatingSchema
from recommender_universal.data.transforms.numerical import MinMaxScaler
from recommender_universal.data.pipeline import DataPipeline
import pytest


def test_pipeline_minmax(tmp_path):
    path = tmp_path / "ratings.csv"
    df_in = pd.DataFrame({"u": [1, 2, 3], "i": [10, 20, 10], "r": [3.0, 5.0, 4.0]})
    df_in.to_csv(path, index=False)

    pipeline = DataPipeline(
        connector=CSVConnector(path),
        schema=RatingSchema(user="u", item="i", rating="r"),
        transforms=[MinMaxScaler(columns=["r"])],
    )

    df_out = pipeline.run()
    assert df_out.shape == df_in.shape
    assert df_out["r"].max() == pytest.approx(1.0)
    assert df_out["r"].min() == pytest.approx(0.0)
