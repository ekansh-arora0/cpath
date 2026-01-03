"""
Visualization Utilities for CPath-Omni

Functions for plotting ROC curves, confusion matrices, attention maps, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: Dict[str, np.ndarray],
    title: str = "ROC Curve",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multiple methods.
    
    Args:
        y_true: Ground truth labels
        y_scores: Dictionary mapping method names to scores
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(y_scores)))
    
    for (method, scores), color in zip(y_scores.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        
        ax.plot(fpr, tpr, color=color, lw=2, 
                label=f'{method} (AUC = {auc:.3f})')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ["Normal", "Tumor"],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "Blues"
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap
    
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def create_attention_overlay(
    image: Union[str, Path, Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet"
) -> np.ndarray:
    """
    Overlay attention heatmap on image.
    
    Args:
        image: Original image
        heatmap: Attention heatmap (values 0-1)
        alpha: Overlay transparency
        colormap: Matplotlib colormap
    
    Returns:
        Overlay image as numpy array
    """
    # Load image
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Resize heatmap if needed
    if heatmap.shape[:2] != image_np.shape[:2]:
        heatmap = np.array(Image.fromarray(heatmap).resize(
            (image_np.shape[1], image_np.shape[0]),
            Image.BILINEAR
        ))
    
    # Apply colormap to heatmap
    cmap = plt.cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]
    
    # Blend
    overlay = (1 - alpha) * image_np + alpha * heatmap_colored
    overlay = (overlay * 255).clip(0, 255).astype(np.uint8)
    
    return overlay


def plot_comparison_grid(
    images: List[Union[str, Path, np.ndarray]],
    titles: List[str],
    nrows: int = 2,
    ncols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a grid of images for comparison.
    
    Args:
        images: List of images
        titles: List of titles
        nrows: Number of rows
        ncols: Number of columns
        figsize: Figure size
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for ax, img, title in zip(axes, images, titles):
        if isinstance(img, (str, Path)):
            img = np.array(Image.open(img))
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    
    # Hide unused axes
    for ax in axes[len(images):]:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ["auc_roc", "accuracy", "f1_score"],
    title: str = "Method Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Bar chart comparing metrics across methods.
    
    Args:
        results: Dictionary mapping method names to metrics
        metrics: List of metrics to compare
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    methods = list(results.keys())
    n_methods = len(methods)
    n_metrics = len(metrics)
    
    x = np.arange(n_methods)
    width = 0.8 / n_metrics
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [results[m].get(metric, 0) * 100 for m in methods]
        offset = width * (i - n_metrics / 2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=metric.replace('_', ' ').title(), color=color)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig


def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: Dict[int, str] = {0: "Normal", 1: "Tumor"},
    title: str = "t-SNE Embedding Visualization",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30
) -> plt.Figure:
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Embedding matrix (N, D)
        labels: Labels for each embedding
        label_names: Mapping from label to name
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        perplexity: t-SNE perplexity parameter
    
    Returns:
        Matplotlib figure
    """
    from sklearn.manifold import TSNE
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        name = label_names.get(label, f"Class {label}")
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                  c=[color], label=name, alpha=0.6, s=20)
    
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return fig
