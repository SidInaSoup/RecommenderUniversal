from .base import register_model, get_model, list_models, get_model_params
from .factory import load_model, load_model_from_config, load_model_from_json
from .decorators import register
from .discover import autoload_models

# Auto-Discover when this module is imported
autoload_models()

__all__ = [
    "register_model",
    "get_model",
    "list_models",
    "get_model_params",
    "load_model",
    "load_model_from_config",
    "load_model_from_json",
    "register",
]
