from .enhanced_cewa import ClassificationCEWA, DetectionCEWA
from .config import Config


def create_aggregator(config: Config):
    """Create aggregator based on configuration."""
    if config.task == 'classification':
        cfg = config.classification
        return ClassificationCEWA(
            n_classes=cfg.n_classes,
            smoothing=cfg.smoothing,
            trust_threshold=cfg.trust_threshold,
            use_majority_vote=cfg.use_majority_vote,
        )
    elif config.task == 'detection':
        cfg = config.detection
        return DetectionCEWA(
            n_classes=cfg.n_classes,
            iou_threshold=cfg.iou_threshold,
            trust_threshold=cfg.trust_threshold,
            use_majority_vote=cfg.use_majority_vote,
            use_recall_weight=cfg.use_recall_weight,
            use_entropy_modulation=cfg.use_entropy_modulation,
            use_spatial_bias=cfg.use_spatial_bias
        )
    else:
        raise ValueError(f"Unknown task: {config.task}")


def get_aggregator(task, **kwargs):
    """Simplified factory for quick creation."""
    if task == 'classification':
        return ClassificationCEWA(**kwargs)
    elif task == 'detection':
        return DetectionCEWA(**kwargs)
    else:
        raise ValueError(f"Unknown task: {task}")