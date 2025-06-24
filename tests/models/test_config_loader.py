import tempfile
import json
import pytest
from recommender_universal.models.registry.factory import (
    load_model_from_config,
    load_model_from_json,
)
from recommender_universal.models.base import BaseRecommender
from recommender_universal.models.baseline import top_popular  # noqa


# Dummy test config for TopPopular model
TOP_POPULAR_CONFIG = {"model": "top_popular", "params": {"item_column": "item_id"}}


def test_load_model_from_config():
    model = load_model_from_config(TOP_POPULAR_CONFIG)
    assert isinstance(model, BaseRecommender)
    assert hasattr(model, "fit")
    assert model.item_column == "item_id"


def test_load_model_from_json():
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
        json.dump(TOP_POPULAR_CONFIG, tmp)
        tmp_path = tmp.name

    model = load_model_from_json(tmp_path)
    assert isinstance(model, BaseRecommender)
    assert model.item_column == "item_id"


def test_invalid_model_name():
    bad_config = {"model": "non_existent_model", "params": {}}
    with pytest.raises(ValueError):
        load_model_from_config(bad_config)


def test_missing_model_key():
    bad_config = {"params": {"item_column": "item_id"}}
    with pytest.raises(ValueError, match="Config must include a 'model' string key"):
        load_model_from_config(bad_config)
