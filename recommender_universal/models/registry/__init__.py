from .base import register_model, get_model, list_models
from .factory import load_model
from .decorators import register
from .discover import autoload_models

# When you import the registry, auto-discover first
autoload_models()

__all__ = [
    "register_model",
    "get_model",
    "list_models",
    "load_model",
    "register",
]
