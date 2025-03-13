"""
Configuration module for Kaggle competition.
Uses Hydra for configuration management.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

# Default config path
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


@dataclass
class PathConfig:
    data_dir: str
    raw_data: str
    processed_data: str
    model_dir: str
    log_dir: str
    submission_dir: str


@dataclass
class FeaturesConfig:
    numerical: List[str]
    categorical: List[str]


@dataclass
class DataConfig:
    train_file: str
    test_file: str
    sample_submission_file: str
    target_column: str
    features: FeaturesConfig
    test_size: float
    random_state: int
    num_folds: int


@dataclass
class FeatureSelectionConfig:
    enabled: bool
    method: str
    k_features: int


@dataclass
class PreprocessingConfig:
    handle_missing: bool
    scaling: str
    categorical_encoding: str
    feature_selection: FeatureSelectionConfig


@dataclass
class AugmentationConfig:
    enabled: bool
    methods: List[str]


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float


@dataclass
class SchedulerConfig:
    name: str
    patience: int
    factor: float
    step_size: int


@dataclass
class EarlyStoppingConfig:
    enabled: bool
    patience: int
    min_delta: float


@dataclass
class GradientClippingConfig:
    enabled: bool
    max_norm: float


@dataclass
class TrainingConfig:
    framework: str
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    batch_size: int
    num_epochs: int
    early_stopping: EarlyStoppingConfig
    gradient_clipping: GradientClippingConfig
    mixed_precision: bool


@dataclass
class ModelConfig:
    name: str
    backbone: str
    pretrained: bool
    dropout: float
    hidden_dim: int
    num_classes: int


@dataclass
class EvaluationConfig:
    metric: str
    monitor: str


@dataclass
class WandbConfig:
    enabled: bool
    project: str
    entity: Optional[str]
    tags: List[str]


@dataclass
class TensorboardConfig:
    enabled: bool


@dataclass
class ConsoleConfig:
    level: str


@dataclass
class LoggingConfig:
    wandb: WandbConfig
    tensorboard: TensorboardConfig
    console: ConsoleConfig


@dataclass
class InferenceConfig:
    batch_size: int
    tta: bool
    checkpoint: str
    ensemble: bool


@dataclass
class ResourcesConfig:
    num_workers: int
    device: str
    pin_memory: bool


@dataclass
class Config:
    paths: PathConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    augmentation: AugmentationConfig
    training: TrainingConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    logging: LoggingConfig
    inference: InferenceConfig
    resources: ResourcesConfig
    seed: int


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def get_config(config_name: str = "default") -> Config:
    """
    Load configuration from yaml files.
    
    Args:
        config_name: Name of the config file (without extension)
        
    Returns:
        Config object
    """
    config_path = os.path.join(CONFIG_PATH, f"{config_name}.yaml")
    config = OmegaConf.load(config_path)
    return config


@hydra.main(config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main function to test configuration loading."""
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main() 