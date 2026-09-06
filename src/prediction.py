"""Funciones para generar predicciones de abandono."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"


def load_model(
    model_path: Path,
) -> Pipeline:
    """Carga un pipeline entrenado."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {model_path}"
        )

    return joblib.load(model_path)


def predict_churn(
    model: Pipeline,
    data: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Genera probabilidades, predicciones y alertas."""

    excluded_columns = [
        column
        for column in [
            ID_COLUMN,
            TARGET_COLUMN,
        ]
        if column in data.columns
    ]

    predictors = data.drop(
        columns=excluded_columns
    )

    probabilities = model.predict_proba(
        predictors
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = pd.DataFrame(
        {
            ID_COLUMN: data[ID_COLUMN],
            "Probabilidad_Churn": probabilities,
            "Prediccion_Churn": predictions,
            "Alerta": np.where(
                predictions == 1,
                "Sí",
                "No",
            ),
        }
    )

    if TARGET_COLUMN in data.columns:
        results["Churn_real"] = (
            data[TARGET_COLUMN].values
        )

    return (
        results
        .sort_values(
            by="Probabilidad_Churn",
            ascending=False,
        )
        .reset_index(drop=True)
    )