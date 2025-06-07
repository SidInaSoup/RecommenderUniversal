import pandas as pd
from recommender_universal.models.registry import load_model

# Sample dataset
df = pd.DataFrame(
    {
        "user_id": [1, 1, 2, 2, 3, 3],
        "item_id": ["A", "B", "A", "C", "B", "D"],
        "rating": [5, 4, 3, 2, 4, 5],
    }
)

# Load the model from registry
model = load_model(
    "mf",
    user_col="user_id",
    item_col="item_id",
    rating_col="rating",
    factors=4,
    epochs=10,
)

# Train model
model.fit(df)

# Generate recommendations for user 1
recs = model.recommend(user_id=1, k=3)
print("Top 3 recommendations for user 1:", recs)
