from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

def EngineeredFeatureField(
        *,
        feature_type: str,
        source_columns: tuple[str, ...],
        **kwargs: Any,
) -> Any:
    return Field(
        ...,
        json_schema_extra={
            "role": "engineered_feature",
            "feature_type": feature_type,
            "source_columns": source_columns,
        },
        **kwargs,
    )

class EngineeredFeature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    BalanceSalaryRatio: float = EngineeredFeatureField(
        feature_type="numeric",
        source_columns=("Balance", "NumOfProducts"),
    )

    TenureAgeRatio: float = EngineeredFeatureField(
        feature_type="numeric",
        source_columns=("Tenure", "Age"),
    )

    IsZeroBalance: int = EngineeredFeatureField(
        feature_type="numeric",
        source_columns=("Balance",),
    )

    EngagementScore: float = EngineeredFeatureField(
        feature_type="numeric",
        source_columns=("HasCrCard", "IsActiveMember")
    )

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        required_columns = {
            column
            for field in cls.model_fields.values()
            for column in (
                field.json_schema_extra or {}
            ).get("source_columns", ())
        }

        missing_columns = required_columns - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns: {sorted(missing_columns)}"
            )
        
        balance = pd.to_numeric(
            dataframe["Balance"],
            errors="coerce"
        )

        salary = pd.to_numeric(
            dataframe["EstimatedSalary"],
            errors="coerce",
        )

        products = pd.to_numeric(
            dataframe["NumOfProducts"],
            errors="coerce",
        )

        tenure = pd.to_numeric(
            dataframe["Tenure"],
            errors="coerce",
        )

        age = pd.to_numeric(
            dataframe["Age"],
            errors="coerce"
        )

        credit_card = pd.to_numeric(
            dataframe["HasCrCard"],
            errors="coerce"
        )

        active = pd.to_numeric(
            dataframe["IsActiveMember"],
            errors="coerce"
        )

        return pd.DataFrame(
            {
                "BalanceSalaryRatio": balance.div(
                    salary.where(salary.ne(0))
                ),
                "BalancePerProduct": balance.div(
                    products.where(products.ne(0))
                ),
                "TenureAgeRatio": tenure.div(
                    age.where(age.ne(0))
                ),
                "IsZeroBalance": balance.eq(0).astype("int8"),
                "EngagementScore": credit_card + active,
            },
            index=dataframe.index,
        )
    
def engineer_features(
        dataframe: pd.DataFrame,
) -> pd.DataFrame:
    engineered = EngineeredFeature.from_dataframe(dataframe)

    return pd.concat(
        [dataframe.copy(), engineered],
        axis=1,
    )

def get_engineered_feature_name(
        feature_type: str | None = None,
) -> tuple[str, ...]:
    return tuple(
        name
        for name, field in EngineeredFeature.model_fields.items()
        if feature_type is None
        or (field.json_schema_extra or {}).get(
            "feature_type"
        )
        == feature_type
    )