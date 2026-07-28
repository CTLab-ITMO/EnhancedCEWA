import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from .enhanced_cewa import EnhancedCEWA, ClassificationCEWA, DetectionCEWA
from .factory import create_aggregator, get_aggregator

__all__ = [
    'EnhancedCEWA',
    'ClassificationCEWA',
    'DetectionCEWA',
    'create_aggregator',
    'get_aggregator',
    'logger',
]