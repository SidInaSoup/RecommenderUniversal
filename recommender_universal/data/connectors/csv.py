import pandas as pd
from .base import BaseConnector


class CSVConnector(BaseConnector):
    """
    Connector for CSV files, automatically registered for
    '.csv' extensions and the 'csv://' URI scheme.
    """

    def _read(self, uri: str, **kwargs) -> pd.DataFrame:
        """
        Load DataFrame from the given URI (local path or URL).
        `kwargs` are passed straight through to pandas.read_csv.
        """
        df = pd.read_csv(uri, **kwargs)
        self._df = df  # for schema introspection
        return df


# Registering this connector under both file-extension and URI-scheme keys:
BaseConnector.register_connector(".csv", CSVConnector)
BaseConnector.register_connector("csv://", CSVConnector)
