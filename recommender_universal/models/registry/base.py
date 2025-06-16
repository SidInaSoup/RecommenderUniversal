from typing import Type, Dict
from recommender_universal.models.base import BaseRecommender

# Internal storage of model name → class
_MODEL_REGISTRY: Dict[str, Type[BaseRecommender]] = {}


def register_model(name: str, model_cls: Type[BaseRecommender]) -> None:
    """Register a model class under the given name."""
    _MODEL_REGISTRY[name] = model_cls


def get_model(name: str) -> Type[BaseRecommender]:
    """Retrieve a registered model class; raise if missing."""
    try:
        return _MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Model '{name}' is not registered.") from None


def list_models() -> list[str]:
    """Return a list of all registered model names."""
    return list(_MODEL_REGISTRY.keys())
