import s3fs
from .base import BaseConnector
import pandas as pd


# pandas on HTTP/S3 auto-detects s3fs if installed,
# so you may not need a dedicated S3Connector.
# But if you want custom handling:
class S3Connector(BaseConnector):
    def _read(self, uri: str, **kwargs) -> pd.DataFrame:
        # e.g. "s3://bucket/path/file.csv"
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(uri, "rb") as f:
            # delegate to CSV or Parquet based on extension
            if uri.endswith(".csv"):
                df = pd.read_csv(f, **kwargs)
            elif uri.endswith(".parquet"):
                df = pd.read_parquet(f, **kwargs)
            else:
                raise ValueError("Unsupported S3 file type")
        self._df = df
        return df


BaseConnector.register_connector("s3://", S3Connector)
