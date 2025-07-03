import pandas as pd
import tempfile  # noqa: F401
from recommender_universal.models.advanced.matrix_factorization import (
    MatrixFactorization,
)


def test_versioned_save_load(tmp_path):
    df = pd.DataFrame(
        {"user_id": [1, 1, 2], "item_id": ["A", "B", "A"], "rating": [5, 4, 3]}
    )
    model = MatrixFactorization(factors=2, epochs=1)
    model.fit(df)

    # Save v1
    model.save(
        base_dir=str(tmp_path), model_name="mf", config={"factors": 2, "epochs": 1}
    )
    # Modify model state and save v2
    model.epochs = 2
    model.save(
        base_dir=str(tmp_path), model_name="mf", config={"factors": 2, "epochs": 2}
    )

    # Load latest (v2)
    loaded2 = MatrixFactorization.load(base_dir=str(tmp_path), model_name="mf")
    assert loaded2.epochs == 2

    # Load v1 explicitly
    loaded1 = MatrixFactorization.load(
        base_dir=str(tmp_path), model_name="mf", version=1
    )
    assert loaded1.epochs == 1
