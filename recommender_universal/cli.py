import argparse
import pandas as pd

from recommender_universal.models.registry import load_model as instantiate_model
from recommender_universal.models.base import BaseRecommender


def train_main():
    parser = argparse.ArgumentParser(description="Train a recommendation model.")
    parser.add_argument(
        "--model", required=True, help="Model name (e.g., top_popular, mf)"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--save-path", required=True, help="Path to save trained model")
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--rating-col", default="rating")
    parser.add_argument("--factors", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)

    args = parser.parse_args()
    df = pd.read_csv(args.input)

    model: BaseRecommender = instantiate_model(
        args.model,
        user_col=args.user_col,
        item_col=args.item_col,
        rating_col=args.rating_col,
        factors=args.factors,
        epochs=args.epochs,
    )
    model.fit(df)
    model.save(args.save_path)
    print(f"✅ Model saved to {args.save_path}")


def predict_main():
    parser = argparse.ArgumentParser(description="Generate recommendations for a user.")
    parser.add_argument(
        "--model", required=True, help="Model name (e.g., top_popular, mf)"
    )
    parser.add_argument("--model-path", required=True, help="Path to saved model file")
    parser.add_argument(
        "--user-id", required=True, type=int, help="User ID to recommend for"
    )
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    # Re-instantiate the right model class from the registry,
    # using dummy constructor args if necessary (we only need self.__dict__)
    model: BaseRecommender = instantiate_model(args.model)
    # Load the saved state (handles both old dict‐pickle and new full‐pickle)
    model.load(args.model_path)

    recs = model.recommend(user_id=args.user_id, k=args.top_k)
    print(f"✅ Top {args.top_k} recommendations for user {args.user_id}: {recs}")


if __name__ == "__main__":
    train_main()
