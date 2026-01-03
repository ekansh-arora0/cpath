"""
Experiment modules for CPath-Omni evaluation
"""

from .exp1_same_cancer import run_same_cancer_experiment
from .exp2_cross_cancer import run_cross_cancer_experiment
from .exp3_cross_species import run_cross_species_experiment
from .gradcam_analysis import GradCAMAnalyzer, generate_gradcam_comparison
from .exp4_gradcam import GradCAMAnalysis, run_gradcam_experiment

__all__ = [
    "run_same_cancer_experiment",
    "run_cross_cancer_experiment", 
    "run_cross_species_experiment",
    "GradCAMAnalyzer",
    "generate_gradcam_comparison",
    "GradCAMAnalysis",
    "run_gradcam_experiment",
]
