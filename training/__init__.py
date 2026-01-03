"""
Training module for CPath-Omni

Includes training utilities for:
- Linear probes
- Few-shot fine-tuning
- Adapter fine-tuning  
- Semantic Anchoring projection head
"""

from .train_utils import (
    LinearProbe,
    FewShotFineTuner,
    AdapterFineTuner,
    PrototypeClassifier,
    run_few_shot_experiment
)

__all__ = [
    "LinearProbe",
    "FewShotFineTuner", 
    "AdapterFineTuner",
    "PrototypeClassifier",
    "run_few_shot_experiment"
]
