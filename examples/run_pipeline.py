import pandas as pd
from pathlib import Path

from recommender_universal.data.connectors.csv import CSVConnector
from recommender_universal.data.schema import RatingSchema
from recommender_universal.data.transforms.numerical import MinMaxScaler
from recommender_universal.data.pipeline import DataPipeline
from recommender_universal.models.baseline.top_popular import TopPopularRecommender
from recommender_universal.evaluation.metrics import hit_rate_at_k

data = {
    "user_id": [1, 2, 1, 3, 2],
    "item_id": [10, 10, 20, 10, 30],
    "rating": [4.0, 5.0, 3.0, 4.5, 2.0],
}

path = Path("examples/test_data/ratings.csv")
path.parent.mkdir(parents=True, exist_ok=True)
df_input = pd.DataFrame(data)
df_input.to_csv(path, index=False)


connector = CSVConnector(path)
schema = RatingSchema(user="user_id", item="item_id", rating="rating")
transforms = [MinMaxScaler(columns=["rating"])]
pipeline = DataPipeline(connector, schema, transforms)

df = pipeline.run()

# Fitting and evaluating
model = TopPopularRecommender().fit(df)
score = hit_rate_at_k(model, df, user_column="user_id", item_column="item_id", k=3)

print(f"Hit Rate at k=3: {score:.4f}")
