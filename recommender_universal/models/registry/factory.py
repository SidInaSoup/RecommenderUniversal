from typing import Any
from recommender_universal.models.base import BaseRecommender
from recommender_universal.models.registry.base import get_model


def load_model(name: str, **kwargs: Any) -> BaseRecommender:
    """
    Instantiate a registered model by name, passing kwargs to its constructor.
    """
    cls = get_model(name)
    return cls(**kwargs)
