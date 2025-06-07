from typing import Type, Dict, Any
from recommender_universal.models.base import BaseRecommender

_MODEL_REGISTRY: Dict[str, Type[BaseRecommender]] = {}


def register_model(name: str, model_cls: Type[BaseRecommender]) -> None:
    _MODEL_REGISTRY[name] = model_cls


def get_model(name: str) -> Type[BaseRecommender]:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is not registered.")
    return _MODEL_REGISTRY[name]


def load_model(name: str, **kwargs: Any) -> BaseRecommender:
    return get_model(name)(**kwargs)


def register(name: str) -> Any:
    def wrapper(cls: Any) -> Any:
        register_model(name, cls)
        return cls

    return wrapper
