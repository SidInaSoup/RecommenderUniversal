import pandas as pd
from recommender_universal.data.transforms.numerical import MinMaxScaler
import pytest


def test_minmax_scaler():
    df = pd.DataFrame({"a": [0, 10, 20], "b": [5, 15, 25]})
    scaler = MinMaxScaler(columns=["a", "b"])
    df_scaled = scaler.fit_transform(df)

    assert df_scaled["a"].min() == pytest.approx(0.0)
    assert df_scaled["a"].max() == pytest.approx(1.0)
    assert df_scaled["b"].min() == pytest.approx(0.0)
    assert df_scaled["b"].max() == pytest.approx(1.0)
