import sys
import pandas as pd
from pathlib import Path  # noqa: F401
import pytest  # noqa: F401

from recommender_universal.cli import train_main, predict_main


def test_cli_train_and_predict(tmp_path, capsys, monkeypatch):
    # Prepare a small training CSV
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "item_id": [10, 20, 10],
            "rating": [4.0, 5.0, 3.0],
        }
    )
    train_csv = tmp_path / "train.csv"
    df.to_csv(train_csv, index=False)

    # Path to save the model
    model_file = tmp_path / "model.pkl"

    # Simulate CLI train command
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model",
            "mf",
            "--input",
            str(train_csv),
            "--save-path",
            str(model_file),
        ],
    )
    train_main()
    out = capsys.readouterr().out
    assert f"✅ Model saved to {model_file}" in out

    # Simulate CLI predict command (now including --model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model",
            "mf",
            "--model-path",
            str(model_file),
            "--user-id",
            "1",
            "--top-k",
            "2",
        ],
    )
    predict_main()
    out = capsys.readouterr().out
    assert "Top 2 recommendations for user 1" in out
