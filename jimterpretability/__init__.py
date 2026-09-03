"""Jimterpretability: mechanistic interpretability platform for open-weight LLMs."""

__version__ = "0.1.0"

from jimterpretability.session import JimSession, SessionConfig, SessionRegistry
from jimterpretability.activations import extract_activations
from jimterpretability.features import aggregate_features, FeatureStats, ConceptDiscoveryResult

__all__ = [
    "JimSession",
    "SessionConfig",
    "SessionRegistry",
    "extract_activations",
    "aggregate_features",
    "FeatureStats",
    "ConceptDiscoveryResult",
]
