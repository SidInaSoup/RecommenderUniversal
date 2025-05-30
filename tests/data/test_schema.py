import pandas as pd
import pytest
from recommender_universal.data.schema import RatingSchema


def test_rating_schema_validates():
    df = pd.DataFrame({"u": [1], "i": [2], "r": [5.0]})
    schema = RatingSchema(user="u", item="i", rating="r")
    schema.validate(df)  # Should not raise


def test_rating_schema_raises():
    df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
    schema = RatingSchema(user="u", item="i", rating="r")
    with pytest.raises(ValueError):
        schema.validate(df)
