"""Script ejecutable para preprocesar el dataset."""

from pathlib import Path
import sys


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from src.data_processing import (  # noqa: E402
    clean_telco_data,
    load_raw_data,
    save_processed_data,
    split_datasets,
)


RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


def main() -> None:
    """Ejecuta el proceso completo de preparación de datos."""

    original_data = load_raw_data(
        RAW_DATA_PATH
    )

    clean_data = clean_telco_data(
        original_data
    )

    (
        train_data,
        validation_data,
        test_data,
    ) = split_datasets(clean_data)

    save_processed_data(
        clean_data=clean_data,
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        output_directory=PROCESSED_DIRECTORY,
    )

    print("Preprocesamiento finalizado.")
    print(f"Base limpia: {len(clean_data)}")
    print(f"TRAIN: {len(train_data)}")
    print(f"VALIDATION: {len(validation_data)}")
    print(f"TEST: {len(test_data)}")


if __name__ == "__main__":
    main()