import pandas as pd
import joblib
import dill
import json
import os  # noqa: F401
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Dict, Any, Optional
from pathlib import Path  # noqa: F401

T = TypeVar("T")


class BaseRecommender(ABC, Generic[T]):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseRecommender":
        pass

    @abstractmethod
    def recommend(self, user_id: int, k: int = 5) -> List[T]:
        """
        Recommend k items for a given user.

        :param user_id: ID of the user to recommend items for.
        :param k: Number of items to recommend.
        :return: List of recommended item IDs.
        """
        pass

    def evaluate(self, test_df: pd.DataFrame) -> float:
        raise NotImplementedError("Evaluation not implemented")

    def save(
        self,
        base_dir: str,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        use_joblib: bool = True,
        use_dill: bool = False,
    ) -> None:
        """
        Save model with versioning and metadata.
        :param base_dir: str, base directory to save the model.
        :param model_name: str, name under which to save (e.g. "mf")
        :param config: dict, dict of constructor params.
        """
        root = Path(base_dir) / model_name
        root.mkdir(parents=True, exist_ok=True)

        # Determine next version
        existing = [
            p.name for p in root.iterdir() if p.is_dir() and p.name.startswith("v")
        ]
        versions = sorted(int(v[1:]) for v in existing if v[1:].isdigit())
        next_version = versions[-1] + 1 if versions else 1
        version_dir = root / f"v{next_version}"
        version_dir.mkdir()

        # Save model binary
        model_path = version_dir / ("model.dill" if use_dill else "model.joblib")
        if use_dill:
            with open(model_path, "wb") as f:
                dill.dump(self, f)
        else:
            joblib.dump(self, model_path)

        # Save metadata & config
        meta = {
            "model_name": model_name,
            "version": next_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": config,
        }
        with open(version_dir / "config.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✅ Saved {model_name} v{next_version} to {version_dir}")

    @classmethod
    def load(
        cls,
        base_dir: str,
        model_name: str,
        version: Optional[int] = None,
        use_dill: bool = False,
    ) -> "BaseRecommender":
        """
        Load a model instance by name and version. If version=None, load latest.
        :param base_dir: str, base directory where models are saved.
        :param model_name: str, name of the model to load (e.g. "mf").
        :param version: int, specific version to load, or None for latest.
        :param use_dill: bool, whether to use dill for loading
                         (default False, otherwise joblib).
        :return: An instance of the model class.
        :raises FileNotFoundError: If the model or version does not exist.
        """
        root = Path(base_dir) / model_name
        if not root.exists():
            raise FileNotFoundError(f"No saved versions for model '{model_name}'")

        # Pick version
        dirs = sorted(
            [d for d in root.iterdir() if d.is_dir() and d.name.startswith("v")],
            key=lambda p: int(p.name[1:]),
        )
        if not dirs:
            raise FileNotFoundError(f"No versions found under {root}")
        if version is None:
            version_dir = dirs[-1]
        else:
            version_dir = root / f"v{version}"
            if not version_dir.exists():
                raise FileNotFoundError(
                    f"Version v{version} not found for '{model_name}'"
                )

        model_path = version_dir / ("model.dill" if use_dill else "model.joblib")
        if use_dill:
            with open(model_path, "rb") as f:
                instance = dill.load(f)
        else:
            instance = joblib.load(model_path)

        return instance
