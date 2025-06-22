import inspect
from typing import Type, Dict
from recommender_universal.models.base import BaseRecommender

# Internal storage of model name → class
_MODEL_REGISTRY: Dict[str, Type[BaseRecommender]] = {}
# Cache of model name → inspect.Signature (without 'self')
_MODEL_SIGNATURES: Dict[str, inspect.Signature] = {}


def register_model(name: str, model_cls: Type[BaseRecommender]) -> None:
    """
    Register a model class under the given name.

    This does not eagerly capture the signature—signatures are
    lazily inspected on first use of get_model_params().
    """
    _MODEL_REGISTRY[name] = model_cls


def get_model(name: str) -> Type[BaseRecommender]:
    """
    Retrieve a registered model class by name.

    Raises:
        ValueError: if the model name is not in the registry.
    """
    try:
        return _MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Model '{name}' is not registered.") from None


def list_models() -> list[str]:
    """
    Return a list of all registered model names.
    """
    return list(_MODEL_REGISTRY.keys())


def get_model_params(name: str) -> inspect.Signature:
    """
    Return the constructor signature for a registered model,
    excluding 'self'. Lazily inspects and caches the Signature
    on first call.

    Raises:
        ValueError: if the model name is not registered.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"No model registered under name '{name}'")

    # If not yet cached, inspect __init__ and cache its signature
    if name not in _MODEL_SIGNATURES:
        cls = _MODEL_REGISTRY[name]
        sig = inspect.signature(cls.__init__)
        # Drop the first parameter ('self')
        params = list(sig.parameters.values())[1:]
        _MODEL_SIGNATURES[name] = sig.replace(parameters=params)

    return _MODEL_SIGNATURES[name]
