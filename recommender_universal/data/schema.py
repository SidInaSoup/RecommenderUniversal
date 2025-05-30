import pandas as pd


class RatingSchema:
    def __init__(
        self, user: str, item: str, rating: str, timestamp: str | None = None
    ) -> None:
        self.user = user
        self.item = item
        self.rating = rating
        self.timestamp = timestamp

    def required_columns(self) -> list[str]:
        cols = [self.user, self.item, self.rating]
        if self.timestamp:
            cols.append(self.timestamp)
        return cols

    def validate(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.required_columns() if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
