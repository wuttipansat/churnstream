from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd
from pydantic import ValidationError

from churnstream.features.config import build_feature_config
from churnstream.schemas.data_schemas import Record

feature_config = build_feature_config()

@dataclass
class ValidationResult:
    is_valid: bool
    row_count: int
    column_count: int
    duplicate_rows: int
    missing_values: dict[str, int] = field(default_factory=dict)
    invalid_rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def invalid_row_count(self) -> int:
        return len(self.invalid_rows)
    
    @property
    def valid_row_count(self) -> int:
        return self.row_count - self.invalid_row_count
    

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_rows": self.duplicate_rows,
            "missing_values": self.missing_values,
            "valid_row_count": self.valid_row_count,
            "invalid_row_count": self.invalid_row_count,
            "invalid_rows": self.invalid_rows,
            "errors": self.errors,
            "warnings": self.warnings,
        }

def validate_data(
        dataframe: pd.DataFrame,
        required_columns: Sequence[str] | None = None,
        target_column: str | None = None,
        validate_rows: bool = True,
        check_target_classes: bool = True,
) -> ValidationResult:
    
    required_columns = required_columns or feature_config.raw_columns
    target_column = target_column or feature_config.target
    
    errors: list[str] = []
    warnings: list[str] = []
    invalid_rows: list[dict[str, Any]] = []

    row_counts = len(dataframe)
    column_count = len(dataframe.columns)

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
        errors.append(
            f"Target column '{target_column}' was not found."
        )
    elif check_target_classes:
        
        target_unique_values = dataframe[target_column].dropna().unique()

        if len(target_unique_values) < 2:
            errors.append(
                f"Target column '{target_column}' contains fewer than two classes."
            )

    if validate_rows and not missing_columns:
        for row_index, row in dataframe.iterrows():
            record = row.where(pd.notna(row), None).to_dict()

            try:
                Record(**record)

            except ValidationError as error:
                invalid_rows.append(
                    {
                        "row_index": int(row_index),
                        "errors": error.errors(),
                        "record": record,
                    }
                )
        
        if invalid_rows:
            errors.append(
                f"{len(invalid_rows)} rows failed schema validation."
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        row_count=row_counts,
        column_count=column_count,
        duplicate_rows=duplicate_rows,
        missing_values=missing_values,
        invalid_rows=invalid_rows,
        errors=errors,
        warnings=warnings,
    )