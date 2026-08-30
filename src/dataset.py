from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.dataset_contract import CHURN_DATASET_CONTRACT
from src.schemas import DatasetRowChurn


class ChurnDataset:
    """Load and validate a churn training dataset from a CSV file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._dataframe: pd.DataFrame | None = None
        self._rows: tuple[DatasetRowChurn, ...] | None = None

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the loaded dataset or fail if ``load`` was not called."""
        if self._dataframe is None:
            raise RuntimeError("Dataset is not loaded")

        return self._dataframe.copy(deep=True)

    def load(self) -> pd.DataFrame:
        """Read the CSV file and verify that its table structure is valid."""
        self._dataframe = None
        self._rows = None
        self._validate_path()

        try:
            dataframe = pd.read_csv(self.path)
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"Dataset file is empty: {self.path}") from error
        except pd.errors.ParserError as error:
            raise ValueError(f"Could not parse CSV file: {self.path}") from error

        self._validate_dataframe(dataframe)
        rows = self._parse_rows(dataframe)

        self._dataframe = dataframe
        self._rows = tuple(rows)
        return self.dataframe

    def to_rows(self) -> list[DatasetRowChurn]:
        if self._rows is None:
            raise RuntimeError("Dataset is not loaded")

        return list(self._rows)

    @staticmethod
    def _parse_rows(dataframe: pd.DataFrame) -> list[DatasetRowChurn]:
        records = dataframe.to_dict(orient="records")
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
        CHURN_DATASET_CONTRACT.validate_columns(dataframe)

        if dataframe.empty:
            raise ValueError("Dataset contains no data rows")
