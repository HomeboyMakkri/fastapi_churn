from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.schemas import DatasetRowChurn


class ChurnDataset:
    """Load and validate a churn training dataset from a CSV file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._dataframe: pd.DataFrame | None = None

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the loaded dataset or fail if ``load`` was not called."""
        if self._dataframe is None:
            raise RuntimeError("Dataset is not loaded")

        return self._dataframe

    def load(self) -> pd.DataFrame:
        """Read the CSV file and verify that its table structure is valid."""
        self._dataframe = None
        self._validate_path()

        try:
            dataframe = pd.read_csv(self.path)
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"Dataset file is empty: {self.path}") from error
        except pd.errors.ParserError as error:
            raise ValueError(f"Could not parse CSV file: {self.path}") from error

        self._validate_dataframe(dataframe)
        self._dataframe = dataframe
        return dataframe

    def to_rows(self) -> list[DatasetRowChurn]:
        records = self.dataframe.to_dict(orient="records")
        rows: list[DatasetRowChurn] = []

        for csv_line, record in enumerate(records, start=2):
            try:
                row = DatasetRowChurn.model_validate(record)
            except ValidationError as error:
                raise ValueError(
                    f"Invalid data at CSV line {csv_line}: {error}"
                ) from error

            rows.append(row)

        return rows

    def _validate_path(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        if not self.path.is_file():
            raise IsADirectoryError(f"Dataset path is not a file: {self.path}")

        if self.path.suffix.lower() != ".csv":
            raise ValueError(f"Dataset file is not a CSV: {self.path}")

    @staticmethod
    def _validate_dataframe(dataframe: pd.DataFrame) -> None:
        if dataframe.empty:
            raise ValueError("Dataset contains no data rows")

        expected_columns = set(DatasetRowChurn.model_fields)
        actual_columns = set(dataframe.columns)
        missing_columns = expected_columns - actual_columns
        unexpected_columns = actual_columns - expected_columns

        if missing_columns or unexpected_columns:
            details: list[str] = []

            if missing_columns:
                details.append(f"missing: {sorted(missing_columns)}")

            if unexpected_columns:
                details.append(f"unexpected: {sorted(unexpected_columns)}")

            raise ValueError(f"Invalid dataset columns ({'; '.join(details)})")
