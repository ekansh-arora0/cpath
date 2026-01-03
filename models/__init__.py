"""
CPath-Omni: Cross-Species Pathology Transfer via Text Anchoring
"""

__version__ = "0.1.0"
__author__ = "PathFoundation"

from .inference import CPathOmniInference
from .cpath_clip import CPathCLIPVisionEncoder
from .text_encoder import QwenTextEncoder, TextProjectionHead
from .semantic_anchoring import (
    SemanticAnchoring,
    SemanticAnchoringConfig,
    SemanticAnchoringTrainer,
    TextProjectionHead as SATextProjectionHead,
    get_default_prompts
)
from .hoptimus import HOptimus0Encoder, HOptimusWithTextAnchoring

__all__ = [
    "CPathOmniInference",
    "CPathCLIPVisionEncoder", 
    "QwenTextEncoder",
    "TextProjectionHead",
    "SemanticAnchoring",
    "SemanticAnchoringConfig",
    "SemanticAnchoringTrainer",
    "get_default_prompts",
    "HOptimus0Encoder",
    "HOptimusWithTextAnchoring",
]
