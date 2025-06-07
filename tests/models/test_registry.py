import recommender_universal.models.baseline.top_popular  # noqa: F401
from recommender_universal.models.registry import load_model


def test_registry_load_top_popular():
    model = load_model("top_popular", item_column="item_id")
    assert model is not None
