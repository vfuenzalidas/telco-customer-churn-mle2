"""Script ejecutable para generar predicciones."""

from pathlib import Path
import json
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

from src.prediction import (  # noqa: E402
    load_model,
    predict_churn,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "modelo_logistico_churn_v1.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "metadata_modelo_v1.json"
)

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "predicciones_script.csv"
)


def main() -> None:
    """Carga el modelo y genera predicciones."""

    model = load_model(
        MODEL_PATH
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as metadata_file:
        metadata = json.load(
            metadata_file
        )

    threshold = float(
        metadata["umbral_operativo"]
    )

    input_data = pd.read_csv(
        INPUT_PATH
    )

    results = predict_churn(
        model=model,
        data=input_data,
        threshold=threshold,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    alert_count = (
        results["Alerta"]
        .eq("Sí")
        .sum()
    )

    print("Predicción finalizada.")
    print(f"Registros evaluados: {len(results)}")
    print(f"Umbral utilizado: {threshold:.4f}")
    print(f"Alertas generadas: {alert_count}")
    print(f"Resultado guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()