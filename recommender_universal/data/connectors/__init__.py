from .base import BaseConnector
from typing import Any, Type
import pandas as pd


def load_data(uri: str, **kwargs: Any) -> pd.DataFrame:
    """
    Convenience function: picks the right connector,
    loads the data, and returns a DataFrame.
    """
    conn_cls: Type[BaseConnector] = BaseConnector.get_connector_for(uri)
    connector = conn_cls()
    return connector.load(uri, **kwargs)
