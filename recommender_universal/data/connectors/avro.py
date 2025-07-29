from fastavro import reader
import pandas as pd
from .base import BaseConnector


class AvroConnector(BaseConnector):
    def load(self, uri: str, **kwargs) -> pd.DataFrame:
        with open(uri, "rb") as fo:
            avro_reader = reader(fo, **kwargs)
            records = [rec for rec in avro_reader]
        df = pd.DataFrame.from_records(records)
        self._df = df
        return df


BaseConnector.register_connector(".avro", AvroConnector)
