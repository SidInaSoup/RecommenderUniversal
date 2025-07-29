import pandas as pd
from .base import BaseConnector


class JSONConnector(BaseConnector):
    def load(self, uri: str, orient: str = None, **kwargs) -> pd.DataFrame:
        df = pd.read_json(uri, orient=orient, **kwargs)
        self._df = df
        return df


BaseConnector.register_connector(".json", JSONConnector)
BaseConnector.register_connector("http://", JSONConnector)
BaseConnector.register_connector("https://", JSONConnector)
