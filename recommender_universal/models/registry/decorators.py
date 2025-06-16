from typing import Type, Callable, Any  # noqa: F401
from recommender_universal.models.base import BaseRecommender
from recommender_universal.models.registry.base import register_model


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
