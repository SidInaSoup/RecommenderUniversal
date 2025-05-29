import pandas as pd
from pathlib import Path
from .base import BaseConnector


class CSVConnector(BaseConnector):
    def __init__(self, path: str | Path, **kwargs):
        """
        Initialize the CSVConnector.

        :param path: Path to the CSV file.
        :param kwargs: Additional keyword arguments for pandas read_csv.
        """
        super().__init__(**kwargs)
        self.path = Path(path)
        self.read_kwargs = kwargs

    def _read(self) -> pd.DataFrame:
        return pd.read_csv(self.path, **self.read_kwargs)
