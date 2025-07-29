import pandas as pd
from .base import BaseConnector
from typing import Optional, Any


class JSONConnector(BaseConnector):
    def _read(
        self, uri: str, orient: Optional[str] = None, **kwargs: Any
    ) -> pd.DataFrame:
        df = pd.read_json(uri, orient=orient, **kwargs)
        self._df = df
        return df


BaseConnector.register_connector(".json", JSONConnector)
BaseConnector.register_connector("http://", JSONConnector)
BaseConnector.register_connector("https://", JSONConnector)
