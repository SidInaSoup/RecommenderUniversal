from typing import Optional, Sequence
import pandas as pd
from .connectors.base import BaseConnector
from .schema import RatingSchema
from .transforms.base import BaseTransform


class DataPipeline:
    def __init__(
        self,
        connector: BaseConnector,
        schema: RatingSchema,
        transforms: Optional[Sequence[BaseTransform]] = None,
    ) -> None:
        self.connector = connector
        self.schema = schema
        self.transforms = transforms or []

    def run(self, validate: bool = True, fit: bool = True) -> pd.DataFrame:
        df = self.connector.load()

        if validate:
            self.schema.validate(df)

        for transform in self.transforms:
            df = transform.fit_transform(df) if fit else transform.transform(df)

        return df
