"""Script ejecutable para entrenar el modelo final."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from src.model_training import (  # noqa: E402
    save_model,
    train_final_model,
)


PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "modelo_logistico_churn_v1.joblib"
)


def main() -> None:
    """Ejecuta el entrenamiento del modelo productivo."""

    train_data = pd.read_csv(
        PROCESSED_DIRECTORY
        / "train.csv"
    )

    validation_data = pd.read_csv(
        PROCESSED_DIRECTORY
        / "validation.csv"
    )

    model = train_final_model(
        train_data=train_data,
        validation_data=validation_data,
    )

    save_model(
        model=model,
        output_path=MODEL_PATH,
    )

    total_training = (
        len(train_data)
        + len(validation_data)
    )

    print("Entrenamiento finalizado.")
    print(
        f"Registros utilizados: {total_training}"
    )
    print(
        f"Modelo guardado en: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()