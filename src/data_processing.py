"""Funciones reutilizables para preparar el dataset Telco Churn."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
RANDOM_STATE = 42


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Carga el dataset original desde un archivo CSV."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {file_path}"
        )

    return pd.read_csv(file_path)


def clean_telco_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Limpia y transforma el dataset Telco Customer Churn."""

    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
        "TotalCharges",
        "SeniorCitizen",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias: {missing_columns}"
        )

    clean_data = dataframe.copy()

    clean_data["TotalCharges"] = pd.to_numeric(
        clean_data["TotalCharges"],
        errors="coerce",
    )

    if clean_data[TARGET_COLUMN].dtype == "object":
        clean_data[TARGET_COLUMN] = (
            clean_data[TARGET_COLUMN]
            .map(
                {
                    "No": 0,
                    "Yes": 1,
                }
            )
        )

    if pd.api.types.is_numeric_dtype(
        clean_data["SeniorCitizen"]
    ):
        clean_data["SeniorCitizen"] = (
            clean_data["SeniorCitizen"]
            .map(
                {
                    0: "No",
                    1: "Yes",
                }
            )
        )

    clean_data = (
        clean_data
        .dropna(
            subset=[
                "TotalCharges",
                TARGET_COLUMN,
            ]
        )
        .drop_duplicates(
            subset=[ID_COLUMN],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return clean_data


def split_datasets(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Divide los datos en TRAIN 70%, VALIDATION 15% y TEST 15%."""

    train_data, temporary_data = train_test_split(
        dataframe,
        test_size=0.30,
        stratify=dataframe[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=0.50,
        stratify=temporary_data[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    return (
        train_data.reset_index(drop=True),
        validation_data.reset_index(drop=True),
        test_data.reset_index(drop=True),
    )


def save_processed_data(
    clean_data: pd.DataFrame,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Guarda los datasets procesados en archivos CSV."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = {
        "telco_churn_limpio.csv": clean_data,
        "train.csv": train_data,
        "validation.csv": validation_data,
        "test.csv": test_data,
    }

    for file_name, dataset in datasets.items():
        output_path = (
            output_directory
            / file_name
        )

        dataset.to_csv(
            output_path,
            index=False,
        )