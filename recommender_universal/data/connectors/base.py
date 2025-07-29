from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Type, Any

# Registry of key → connector class
# Keys might be file extensions ('.csv') or schemes ('sqlite://')
_CONNECTOR_REGISTRY: Dict[str, Type["BaseConnector"]] = {}


class BaseConnector(ABC):
    """
    Abstract base class for all data connectors.
    Subclasses implement `_read(uri, **kwargs)` to actually load data.
    """

    def __init__(self, cache: bool = True):
        """
        :param cache: If True, subsequent loads return a cached DataFrame.
        """
        self._cache = cache
        self._cached_df: pd.DataFrame | None = None

    @abstractmethod
    def _read(self, uri: str, **kwargs: Any) -> pd.DataFrame:
        """
        Subclasses implement this to load a DataFrame from `uri`.
        """
        ...

    def load(self, uri: str, refresh: bool = False, **kwargs: Any) -> pd.DataFrame:
        """
        Load data from `uri`. Honors cache unless `refresh=True`.

        :param uri: Data source identifier (file path, URL, DB URI).
        :param refresh: If True, ignore cache and re-run `_read`.
        :param kwargs: Passed through to `_read` (e.g. table_name, engine args).
        :return: Loaded DataFrame.
        """
        if self._cache and self._cached_df is not None and not refresh:
            return self._cached_df

        df = self._read(uri, **kwargs)
        if self._cache:
            self._cached_df = df
        return df

    @property
    def schema(self) -> Dict[str, str]:
        """
        After loading, inspect column types.

        :return: Mapping column_name → dtype (as string).
        """
        if self._cached_df is None:
            raise RuntimeError("No data loaded yet; call `.load()` first.")
        return {col: str(dtype) for col, dtype in self._cached_df.dtypes.items()}

    @classmethod
    def register_connector(cls, key: str, connector_cls: Type["BaseConnector"]) -> None:
        """
        Register a connector under `key`.
        E.g. '.csv', 'sqlite://', 'http://', 's3://'
        """
        _CONNECTOR_REGISTRY[key] = connector_cls

    @classmethod
    def get_connector_for(cls, uri: str) -> Type["BaseConnector"]:
        """
        Choose a connector based on URI scheme or extension.
        """
        for key, connector in _CONNECTOR_REGISTRY.items():
            if uri.startswith(f"{key}") or uri.endswith(key):
                return connector
        raise ValueError(f"No connector found for URI '{uri}'")
