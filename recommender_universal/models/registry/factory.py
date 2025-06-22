import json
from typing import Any, Dict
from recommender_universal.models.base import BaseRecommender
from recommender_universal.models.registry.base import get_model, get_model_params


def load_model(name: str, **kwargs: Any) -> BaseRecommender:
    """
    Instantiate a registered model by name, passing kwargs to its constructor.
    Validates that kwargs match the model's constructor signature
    """
    sig = get_model_params(name)
    for k in kwargs:
        if k not in sig.parameters:
            raise TypeError(f"Unexpected paramter '{k}' for model '{name}'")
    cls = get_model(name)
    return cls(**kwargs)


def load_model_from_config(config: Dict[str, Any]) -> BaseRecommender:
    """
    Load model from a config dict, e.g. {"model": "mf", "params": {...}}.
    """
    name = config.get("model")
    if not isinstance(name, str):
        raise ValueError("Config must include a 'model' string key")
    params = config.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("'params' must be a dict of init arguments")
    return load_model(name, **params)


def load_model_from_json(path: str) -> BaseRecommender:
    """
    Load model config from a JSON file, then instantiate.
    """
    with open(path) as f:
        config = json.load(f)
    return load_model_from_config(config)
