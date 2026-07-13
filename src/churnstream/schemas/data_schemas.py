from typing import Literal, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator

def FeatureField(*, feature_type: str, **kwargs: Any) -> Any:
    return Field(
        ...,
        json_schema_extra={
            "role": "feature",
            "feature_type": feature_type,
        },
        **kwargs,
    )

def IdentifierField(**kwargs: Any) -> Any:
    return Field(
        ...,
        json_schema_extra={
            "role": "identifier",
        },
        **kwargs,
    )

def TargetField(**kwargs: Any) -> Any:
    return Field(
        ...,
        json_schema_extra={
            "role": "target",
        },
        **kwargs,
    )

class FeatureModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    CreditScore: int = FeatureField(feature_type="numeric", ge=0)
    Geography: Literal["France", "Spain", "Germany"] = FeatureField(feature_type="categorical")
    Gender: Literal["Male", "Female"] = FeatureField(feature_type="categorical")

    Age: int = FeatureField(feature_type="numeric", ge=18)
    Tenure: int = FeatureField(feature_type="numeric", ge=0)
    Balance: float = FeatureField(feature_type="numeric", ge=0)

    NumOfProducts: int = FeatureField(feature_type="numeric", ge=0)
    HasCrCard: Literal[0, 1] = FeatureField(feature_type="numeric")
    IsActiveMember: Literal[0, 1] = FeatureField(feature_type="numeric")

    EstimatedSalary: float = FeatureField(feature_type="numeric", ge=0)

class Record(FeatureModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    id: Optional[str] = Field(default=None, alias="_id")

    RowNumber: int = IdentifierField(ge=1)
    CustomerId: int = IdentifierField(ge=1)
    Surname: str = IdentifierField(min_length=1)

    Exited: Literal[0, 1] = TargetField()

    @field_validator("id", mode="before")
    @classmethod
    def convert_object_id_to_string(cls, value: Any) -> Optional[str]:
        return None if value is None else str(value)
    
class PredictionInput(FeatureModel):
    pass

class PredictionOutput(BaseModel):
    prediction: Literal[0, 1]
    probability: float = Field(..., ge=0, le=1)

