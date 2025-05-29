from abc import ABC, abstractmethod
import pandas as pd


class BaseConnector(ABC):
    """
    Abstract base class for all data connectors.
    """

    def __init__(self, cache: bool = True):
        """
        Initialize the BaseConnector.

        :param cache: Whether to cache the data.
        """
        self._cache = cache
        self._cached_df = None

    @abstractmethod
    def _read(self) -> pd.DataFrame:
        """Subclasses must implement the actual data loading logic."""
        pass

    def load(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load the data, either from cache or by reading it.

        :param refresh: If True, force a reload of the data.
        :return: DataFrame containing the loaded data.
        """
        if self._cache and self._cached_df is not None and not refresh:
            return self._cached_df
        df = self._read()  # protected to allow subclasses to override this method
        if self._cache:
            self._cached_df = df
        return df
