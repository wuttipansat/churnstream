from dataclasses import dataclass

from churnstream.schemas.data_schemas import Record

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
        return self.model_categorical + self.engineered_categorical
    
    @property
    def model_features(self) -> tuple[str, ...]:
        return self.model_numeric + self.model_categorical
    
    @property
    def training_columns(self) -> tuple[str, ...]:
        return self.model_features + (self.target,)
    
    @property
    def raw_columns(self) -> tuple[str, ...]:
        return self.identifiers + self.input_features + (self.target,)
    
def build_feature_config() -> FeatureConfig:
    identifiers: list[str] = []
    numeric: list[str] = []
    categorical: list[str] = []
    targets: list[str] = []

    for field_name, field_info in Record.model_fields.items():
        metadata = field_info.json_schema_extra or {}

        role = metadata.get("role")
        feature_type = metadata.get("feature_type")

        if role == "identifier":
            identifiers.append(field_name)

        elif role == "target":
            targets.append(field_name)

        elif role == "feature":
            if feature_type == "numeric":
                numeric.append(field_name)
            elif feature_type == "categorical":
                categorical.append(field_name)

    if len(targets) != 1:
        raise ValueError(f"Expected exactly one target column, got: {targets}")
    
    return FeatureConfig(
        target=targets[0],
        identifiers=tuple(identifiers),
        numeric=tuple(numeric),
        categorical=tuple(categorical),
    )