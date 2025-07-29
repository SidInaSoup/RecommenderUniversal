from fastavro import reader
import pandas as pd
from .base import BaseConnector
from typing import Any


class AvroConnector(BaseConnector):
    def _read(self, uri: str, **kwargs: Any) -> pd.DataFrame:
        with open(uri, "rb") as fo:
            avro_reader = reader(fo, **kwargs)
            records = [rec for rec in avro_reader]
        df = pd.DataFrame.from_records(records)
        self._df = df
        return df


BaseConnector.register_connector(".avro", AvroConnector)
