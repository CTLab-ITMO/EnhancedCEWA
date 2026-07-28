"""Configuration management for EnhancedCEWA."""
from pathlib import Path

import yaml
import logging
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel, ValidationError

from src.utils import Singleton

logger = logging.getLogger(__name__)


@dataclass
class ClassificationConfig:
    n_classes: int = 10
    smoothing: float = 1e-5
    trust_threshold: float = 0.999
    use_majority_vote: bool = False
    predictions_path: Optional[str] = None


@dataclass
class DetectionConfig:
    n_classes: int = 14
    iou_threshold: float = 0.5
    trust_threshold: float = 0.999
    use_majority_vote: bool = False
    use_recall_weight: bool = True
    use_entropy_modulation: bool = True
    use_spatial_bias: bool = True
    predictions_path: Optional[str] = None


class Config(Singleton, BaseModel):
    task: str = 'classification'
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    output_dir: str = 'output'
    verbose: bool = True
    annotations_path: str = None

    @classmethod
    def load(cls, path: Path):
        with open(str(path), "r", encoding='utf-8') as stream:
            try:
                config = yaml.safe_load(stream)
                cls._instance = cls(from_file=True, **config)
            except yaml.YAMLError as e:
                raise RuntimeError(f"Error loading configuration file: {e}")