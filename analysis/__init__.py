"""
Analysis module for CPath-Omni

Tools for analyzing embedding spaces, class separation, and model behavior.
"""

from .embedding_analysis import (
    compute_prototype_similarity,
    create_similarity_heatmap,
    visualize_embedding_space,
    analyze_class_separation,
    compare_models,
    plot_similarity_distribution
)

__all__ = [
    "compute_prototype_similarity",
    "create_similarity_heatmap",
    "visualize_embedding_space",
    "analyze_class_separation",
    "compare_models",
    "plot_similarity_distribution"
]
