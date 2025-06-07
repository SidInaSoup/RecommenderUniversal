import pkgutil
import importlib
from typing import Type, Dict, Any, Callable
from recommender_universal.models.base import BaseRecommender

# --------------------------------------------------------------------------------
# Model registry storage
# --------------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, Type[BaseRecommender]] = {}


def register_model(name: str, model_cls: Type[BaseRecommender]) -> None:
    """Register a BaseRecommender subclass under the given name."""
    _MODEL_REGISTRY[name] = model_cls


def get_model(name: str) -> Type[BaseRecommender]:
    """
    Retrieve the class for a registered model.

    Raises:
        ValueError if no model under that name exists.
    """
    try:
        return _MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Model '{name}' is not registered.") from None


def load_model(name: str, **kwargs: Any) -> BaseRecommender:
    """
    Instantiate a registered model by name, passing kwargs to its constructor.
    """
    model_cls = get_model(name)
    return model_cls(**kwargs)


# --------------------------------------------------------------------------------
# Decorator API
# --------------------------------------------------------------------------------


def register(name: str) -> Callable[[Type[BaseRecommender]], Type[BaseRecommender]]:
    """
    Class decorator to register a model under `name`.

    Usage:
        @register("top_popular")
        class TopPopularRecommender(BaseRecommender):
            ...
    """

    def wrapper(cls: Type[BaseRecommender]) -> Type[BaseRecommender]:
        register_model(name, cls)
        return cls

    return wrapper


# --------------------------------------------------------------------------------
# Auto-load all submodules in recommender_universal.models
# --------------------------------------------------------------------------------


def autoload_models() -> None:
    """
    Walk through the `recommender_universal.models` package and import every
    submodule so that any @register decorators execute.
    """
    import recommender_universal.models  # parent package

    package = recommender_universal.models

    for finder, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        if not is_pkg:
            importlib.import_module(module_name)


# Trigger auto-discovery on import
autoload_models()
