"""Funciones reutilizables para entrenar el modelo de churn."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42


def build_model_pipeline(
    predictors: pd.DataFrame,
) -> Pipeline:
    """Construye el pipeline de preprocesamiento y clasificación."""

    numeric_columns = (
        predictors
        .select_dtypes(include=["number"])
        .columns
        .tolist()
    )

    categorical_columns = [
        column
        for column in predictors.columns
        if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputation",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaling",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputation",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    l1_ratio=0.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model_pipeline


def train_final_model(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
) -> Pipeline:
    """Entrena el modelo final usando TRAIN y VALIDATION."""

    final_training_data = pd.concat(
        [
            train_data,
            validation_data,
        ],
        ignore_index=True,
    )

    predictors = final_training_data.drop(
        columns=[
            ID_COLUMN,
            TARGET_COLUMN,
        ]
    )

    target = final_training_data[
        TARGET_COLUMN
    ]

    model_pipeline = build_model_pipeline(
        predictors
    )

    model_pipeline.fit(
        predictors,
        target,
    )

    return model_pipeline


def save_model(
    model: Pipeline,
    output_path: Path,
) -> None:
    """Guarda el pipeline entrenado."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path,
    )