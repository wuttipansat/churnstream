from dataclasses import dataclass

from pydantic import BaseModel

from churnstream.schemas.data_schemas import Record
from churnstream.features.engineering import EngineeredFeature

@dataclass(frozen=True)
class FeatureConfig:
    target: str
    identifiers: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    engineered_numeric: tuple[str, ...] = ()
    engineered_categorical: tuple[str, ...] = ()

    @property
    def input_features(self) -> tuple[str, ...]:
        return self.numeric + self.categorical
    
    @property
    def model_numeric(self) -> tuple[str, ...]:
        return self.numeric + self.engineered_numeric
    
    @property
    def model_categorical(self) -> tuple[str, ...]:
        return self.categorical + self.engineered_categorical
    
    @property
    def model_features(self) -> tuple[str, ...]:
        return self.model_numeric + self.model_categorical
    
    @property
    def training_columns(self) -> tuple[str, ...]:
        return self.model_features + (self.target,)
    
    @property
    def raw_columns(self) -> tuple[str, ...]:
        return self.identifiers + self.input_features + (self.target,)
    
def get_fields(
        model: type[BaseModel],
        role: str,
        feature_type: str | None = None,
) -> tuple[str, ...]:
    return tuple(
        name
        for name, field in model.model_fields.items()
        if (field.json_schema_extra or {}).get("role") == role
        and (
            feature_type is None
            or (field.json_schema_extra or {}).get("feature_type")
            == feature_type
        )
    )

def build_feature_config() -> FeatureConfig:

    targets = get_fields(Record, "target")

    if len(targets) != 1:
        raise ValueError(
            f"Expected exactly one target column, got: {targets}"
        )
    
    return FeatureConfig(
        target=targets[0],
        identifiers=get_fields(Record, "identifier"),
        numeric=get_fields(Record, "feature", "numeric"),
        categorical=get_fields(Record, "feature", "categorical"),
        engineered_numeric=get_fields(EngineeredFeature, "engineered_feature", "numeric"),
        engineered_categorical=get_fields(EngineeredFeature, "engineered_feature", "categorical")
    )