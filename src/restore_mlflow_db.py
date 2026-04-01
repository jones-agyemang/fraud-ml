from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


REGISTERED_MODEL_NAME = "fraud-detection-model"
EXPERIMENT_NAME = "fraud-detection"


@dataclass(order=True)
class SavedModel:
    created_at: str
    model_id: str
    run_id: str
    model_dir: Path


def parse_mlmodel(model_file: Path) -> SavedModel:
    values: dict[str, str] = {}
    for raw_line in model_file.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith(("model_id:", "run_id:", "utc_time_created:")):
            key, value = line.split(":", 1)
            values[key] = value.strip().strip("'").strip('"')

    missing = {"model_id", "run_id", "utc_time_created"} - values.keys()
    if missing:
        raise ValueError(f"{model_file} is missing fields: {sorted(missing)}")

    return SavedModel(
        created_at=values["utc_time_created"],
        model_id=values["model_id"],
        run_id=values["run_id"],
        model_dir=model_file.parent,
    )


def list_saved_models(models_root: Path) -> list[SavedModel]:
    model_files = sorted(models_root.glob("*/artifacts/MLmodel"))
    models = [parse_mlmodel(model_file) for model_file in model_files]
    return sorted(models)


def ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)


def restore_registry(db_path: Path, models_root: Path) -> None:
    tracking_uri = f"sqlite:///{db_path.resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    mlflow.set_experiment(EXPERIMENT_NAME)
    ensure_registered_model(client, REGISTERED_MODEL_NAME)

    restored_versions = []
    for saved_model in list_saved_models(models_root):
        model_source = saved_model.model_dir.resolve().as_uri()
        version = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=model_source,
            run_id=saved_model.run_id,
            description=(
                f"Restored from saved MLflow model artifact {saved_model.model_id} "
                f"created at {saved_model.created_at}"
            ),
        )
        restored_versions.append((saved_model, version.version))
        print(
            "restored",
            f"version={version.version}",
            f"run_id={saved_model.run_id}",
            f"source={saved_model.model_dir}",
        )

    if restored_versions:
        latest_version = restored_versions[-1][1]
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=latest_version,
        )
        print(f"alias champion -> version {latest_version}")
    else:
        print("no saved models found to restore")

    print(f"tracking_uri={tracking_uri}")
    print(f"registered_model={REGISTERED_MODEL_NAME}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild a fresh MLflow SQLite backend from saved model artifacts."
    )
    parser.add_argument(
        "--db-path",
        default="mlflow.db",
        help="Path to the SQLite backend store to create or update.",
    )
    parser.add_argument(
        "--models-root",
        default="mlruns/1/models",
        help="Root directory containing saved MLflow packaged models.",
    )
    args = parser.parse_args()

    restore_registry(
        db_path=Path(args.db_path),
        models_root=Path(args.models_root),
    )


if __name__ == "__main__":
    main()
