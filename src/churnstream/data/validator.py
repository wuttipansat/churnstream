from dataclasses import dataclass, field
from typing import Any

import pandas as pd

@dataclass
class ValidationResult:
    is_valid: bool
    row_count: int
    column_count: int
    duplicate_rows: int
    missing_values: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_rows": self.duplicate_rows,
            "missing_values": self.missing_values,
            "errors": self.errors,
            "warnings": self.warnings,
        }

def validate_data(
        dataframe: pd.DataFrame,
        required_columns: list[str] | None = None,
        target_column: str = "Exited",
) -> ValidationResult:
    
    errors: list[str] = []
    warnings: list[str] = []

    if dataframe.empty:
        errors.append("The dataset is empty.")

    duplicate_rows = int(dataframe.duplicated().sum())

    if duplicate_rows > 0:
        warnings.append(
            f"Dataset contains {duplicate_rows} duplicated rows."
        )

    missing_values = {
        column: int(count)
        for column, count in dataframe.isnull().sum().items()
        if count > 0
    }

    if missing_values:
        warnings.append("Dataset contains missing values.")

    if required_columns:
        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            errors.append(
                f"Missing required columns: {missing_columns}"
            )

    if target_column not in dataframe.columns:
        warnings.append(
            f"Target column '{target_column}' was not found."
        )

    else:
        target_uniques_values = dataframe[target_column].dropna().unique()

        if len(target_uniques_values) < 2:
            errors.append(
                f"Target column '{target_column}' contains fewer than two classes."
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        duplicate_rows=duplicate_rows,
        missing_values=missing_values,
        errors=errors,
        warnings=warnings,
    )