import pkgutil
import importlib


def autoload_models() -> None:
    """
    Walk through all submodules under `recommender_universal.models`
    and import them so that any @register decorators execute.
    """
    import recommender_universal.models  # noqa: F401

    package = recommender_universal.models

    for finder, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        if not is_pkg:
            importlib.import_module(module_name)
