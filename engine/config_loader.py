try:
    import tomllib
except ImportError:
    import tomli as tomllib

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    path: str
    format: str = "csv"
    separator: str = ","
    drop_columns: List[str] = field(default_factory=list)


@dataclass
class ColumnsConfig:
    target: str
    task: str
    continuous: List[str] = field(default_factory=list)
    discrete: List[str] = field(default_factory=list)


@dataclass
class AttributesConfig:
    key: List[str] = field(default_factory=list)
    sensitive: List[str] = field(default_factory=list)


@dataclass
class PreprocessingConfig:
    date_columns: List[str] = field(default_factory=list)
    pca: bool = False
    pca_variance: float = 0.9
    pca_exclude: List[str] = field(default_factory=list)
    missing_noise_std: float = 0.05


@dataclass
class TrainingConfig:
    gms: List[str] = field(default_factory=lambda: ["CTGAN"])
    losses: List[str] = field(default_factory=lambda: ["vanilla"])
    epochs: int = 10000
    batch_size: int = 500


@dataclass
class DifferentialPrivacyConfig:
    enabled: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    dp_sigma: float = 1.0
    weight_clip: float = 0.1


@dataclass
class PostprocessingConfig:
    constraints: List[str] = field(default_factory=list)


@dataclass
class OutputConfig:
    n_samples: Optional[int] = None
    output_dir: str = "output/{dataset_name}/synthetic/"
    report: bool = False


@dataclass
class PipelineConfig:
    data: DataConfig
    columns: ColumnsConfig
    attributes: AttributesConfig = field(default_factory=AttributesConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    differential_privacy: DifferentialPrivacyConfig = field(
        default_factory=DifferentialPrivacyConfig
    )
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str) -> PipelineConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    if "data" not in raw:
        raise ValueError(f"Config missing required [data] section: {path}")
    if "columns" not in raw:
        raise ValueError(f"Config missing required [columns] section: {path}")

    col_raw = raw["columns"]
    for required in ("target", "task"):
        if required not in col_raw:
            raise ValueError(f"Config missing required [columns].{required}: {path}")

    return PipelineConfig(
        data=DataConfig(**raw["data"]),
        columns=ColumnsConfig(**col_raw),
        attributes=AttributesConfig(**raw.get("attributes", {})),
        preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
        training=TrainingConfig(**raw.get("training", {})),
        differential_privacy=DifferentialPrivacyConfig(
            **raw.get("differential_privacy", {})
        ),
        postprocessing=PostprocessingConfig(**raw.get("postprocessing", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
