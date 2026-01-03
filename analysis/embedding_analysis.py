"""
Embedding Analysis Module

Tools for analyzing the embedding space of pathology foundation models:
- Cosine similarity heatmaps
- Prototype analysis
- t-SNE visualization
- Class separation metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple
import torch


def compute_prototype_similarity(
    embeddings: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Compute cosine similarity between class prototypes.
    
    High similarity (>0.95) indicates poor class separation in embedding space.
    
    Args:
        embeddings: [N, D] feature vectors
        labels: [N] binary labels (0=normal, 1=tumor)
        normalize: Whether to L2-normalize embeddings
        
    Returns:
        Dictionary with prototype similarities and statistics
    """
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
    
    tumor_mask = labels == 1
    normal_mask = labels == 0
    
    tumor_emb = embeddings[tumor_mask]
    normal_emb = embeddings[normal_mask]
    
    # Compute prototypes (mean embeddings)
    tumor_proto = tumor_emb.mean(axis=0)
    normal_proto = normal_emb.mean(axis=0)
    
    # Normalize prototypes
    tumor_proto = tumor_proto / np.linalg.norm(tumor_proto)
    normal_proto = normal_proto / np.linalg.norm(normal_proto)
    
    # Prototype-to-prototype similarity
    proto_similarity = float(tumor_proto @ normal_proto)
    
    # Intra-class similarities
    tumor_intra = cosine_similarity(tumor_emb).mean()
    normal_intra = cosine_similarity(normal_emb).mean()
    
    # Inter-class similarity
    inter_sim = cosine_similarity(tumor_emb, normal_emb).mean()
    
    return {
        "prototype_cosine_similarity": proto_similarity,
        "tumor_intra_class_similarity": float(tumor_intra),
        "normal_intra_class_similarity": float(normal_intra),
        "inter_class_similarity": float(inter_sim),
        "separation_ratio": float((tumor_intra + normal_intra) / 2 - inter_sim)
    }


def create_similarity_heatmap(
    embeddings_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> np.ndarray:
    """
    Create cosine similarity heatmap between prototypes across domains.
    
    Args:
        embeddings_dict: Dictionary mapping domain names to (embeddings, labels) tuples
        output_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Similarity matrix as numpy array
    """
    # Compute prototypes for each domain
    prototypes = {}
    labels_list = []
    
    for domain, (emb, labels) in embeddings_dict.items():
        # Normalize
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        
        tumor_proto = emb[labels == 1].mean(axis=0)
        normal_proto = emb[labels == 0].mean(axis=0)
        
        tumor_proto = tumor_proto / np.linalg.norm(tumor_proto)
        normal_proto = normal_proto / np.linalg.norm(normal_proto)
        
        prototypes[f"{domain}_tumor"] = tumor_proto
        prototypes[f"{domain}_normal"] = normal_proto
        labels_list.extend([f"{domain}_tumor", f"{domain}_normal"])
    
    # Compute pairwise similarities
    n = len(prototypes)
    similarity_matrix = np.zeros((n, n))
    
    for i, name_i in enumerate(labels_list):
        for j, name_j in enumerate(labels_list):
            similarity_matrix[i, j] = prototypes[name_i] @ prototypes[name_j]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=0.9, vmax=1.0)
    
    # Labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels_list, rotation=45, ha='right')
    ax.set_yticklabels(labels_list)
    
    # Add values
    for i in range(n):
        for j in range(n):
            color = 'white' if similarity_matrix[i, j] > 0.95 else 'black'
            ax.text(j, i, f'{similarity_matrix[i, j]:.3f}',
                   ha='center', va='center', color=color, fontsize=8)
    
    plt.colorbar(im, label='Cosine Similarity')
    ax.set_title('Prototype Cosine Similarity Matrix', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()
    
    return similarity_matrix


def visualize_embedding_space(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    domain_labels: Optional[np.ndarray] = None,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30,
    n_components: int = 2
) -> np.ndarray:
    """
    Visualize embedding space using t-SNE or PCA.
    
    Args:
        embeddings: [N, D] feature vectors
        labels: [N] class labels
        method: "tsne" or "pca"
        domain_labels: Optional domain labels for coloring
        output_path: Path to save figure
        figsize: Figure size
        perplexity: t-SNE perplexity parameter
        n_components: Number of components for visualization
        
    Returns:
        2D coordinates from dimensionality reduction
    """
    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Dimensionality reduction
    if method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000
        )
    else:
        reducer = PCA(n_components=n_components, random_state=42)
    
    coords = reducer.fit_transform(embeddings)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=figsize)
    
    if domain_labels is not None:
        # Color by domain with shape by class
        unique_domains = np.unique(domain_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_domains)))
        
        for i, domain in enumerate(unique_domains):
            domain_mask = domain_labels == domain
            
            # Tumor
            mask = domain_mask & (labels == 1)
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=[colors[i]], marker='o', s=30, alpha=0.6,
                      label=f'{domain} Tumor')
            
            # Normal
            mask = domain_mask & (labels == 0)
            ax.scatter(coords[mask, 0], coords[mask, 1],
                      c=[colors[i]], marker='x', s=30, alpha=0.6,
                      label=f'{domain} Normal')
    else:
        # Simple tumor vs normal coloring
        ax.scatter(coords[labels == 1, 0], coords[labels == 1, 1],
                  c='red', marker='o', s=30, alpha=0.6, label='Tumor')
        ax.scatter(coords[labels == 0, 0], coords[labels == 0, 1],
                  c='blue', marker='x', s=30, alpha=0.6, label='Normal')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel(f'{method.upper()} 1')
    ax.set_ylabel(f'{method.upper()} 2')
    ax.set_title(f'Embedding Space Visualization ({method.upper()})', fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()
    
    return coords


def analyze_class_separation(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Comprehensive analysis of class separation in embedding space.
    
    Args:
        embeddings: [N, D] feature vectors
        labels: [N] binary labels
        
    Returns:
        Dictionary with various separation metrics
    """
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    tumor_emb = embeddings[labels == 1]
    normal_emb = embeddings[labels == 0]
    
    # Prototypes
    tumor_proto = tumor_emb.mean(axis=0)
    normal_proto = normal_emb.mean(axis=0)
    
    tumor_proto = tumor_proto / np.linalg.norm(tumor_proto)
    normal_proto = normal_proto / np.linalg.norm(normal_proto)
    
    # Metrics
    results = {}
    
    # 1. Prototype similarity
    results["prototype_similarity"] = float(tumor_proto @ normal_proto)
    
    # 2. Average similarity to own prototype
    results["tumor_to_tumor_proto"] = float((tumor_emb @ tumor_proto).mean())
    results["normal_to_normal_proto"] = float((normal_emb @ normal_proto).mean())
    
    # 3. Average similarity to other prototype
    results["tumor_to_normal_proto"] = float((tumor_emb @ normal_proto).mean())
    results["normal_to_tumor_proto"] = float((normal_emb @ tumor_proto).mean())
    
    # 4. Separability index
    # How much closer are samples to their own prototype vs other
    tumor_margin = results["tumor_to_tumor_proto"] - results["tumor_to_normal_proto"]
    normal_margin = results["normal_to_normal_proto"] - results["normal_to_tumor_proto"]
    results["average_margin"] = (tumor_margin + normal_margin) / 2
    
    # 5. Fisher's criterion (between-class vs within-class variance ratio)
    tumor_var = np.var(tumor_emb @ tumor_proto)
    normal_var = np.var(normal_emb @ normal_proto)
    between_class = ((tumor_proto - normal_proto) ** 2).sum()
    results["fisher_criterion"] = float(between_class / (tumor_var + normal_var + 1e-8))
    
    # 6. Silhouette-like score
    # Average (intra-class sim - inter-class sim)
    intra_tumor = cosine_similarity(tumor_emb).mean()
    intra_normal = cosine_similarity(normal_emb).mean()
    inter = cosine_similarity(tumor_emb, normal_emb).mean()
    
    results["intra_class_similarity"] = float((intra_tumor + intra_normal) / 2)
    results["inter_class_similarity"] = float(inter)
    results["silhouette_like"] = results["intra_class_similarity"] - results["inter_class_similarity"]
    
    return results


def compare_models(
    model_embeddings: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare class separation across different models.
    
    Args:
        model_embeddings: Dict mapping model names to (embeddings, labels)
        output_path: Path to save comparison figure
        
    Returns:
        Dictionary with metrics for each model
    """
    results = {}
    
    for model_name, (emb, labels) in model_embeddings.items():
        results[model_name] = analyze_class_separation(emb, labels)
    
    # Create comparison bar chart
    if output_path:
        metrics = ["prototype_similarity", "average_margin", "silhouette_like"]
        model_names = list(results.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [results[m][metric] for m in model_names]
            axes[i].bar(model_names, values)
            axes[i].set_title(metric.replace("_", " ").title())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    return results


def plot_similarity_distribution(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot distribution of similarities to each prototype.
    
    This visualization shows why prototype-based classification
    may fail when distributions overlap significantly.
    """
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Compute prototypes
    tumor_proto = embeddings[labels == 1].mean(axis=0)
    normal_proto = embeddings[labels == 0].mean(axis=0)
    
    tumor_proto = tumor_proto / np.linalg.norm(tumor_proto)
    normal_proto = normal_proto / np.linalg.norm(normal_proto)
    
    # Compute similarities
    tumor_to_tumor = embeddings[labels == 1] @ tumor_proto
    tumor_to_normal = embeddings[labels == 1] @ normal_proto
    normal_to_tumor = embeddings[labels == 0] @ tumor_proto
    normal_to_normal = embeddings[labels == 0] @ normal_proto
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Similarity to tumor prototype
    axes[0].hist(tumor_to_tumor, bins=50, alpha=0.5, label='Tumor samples', color='red')
    axes[0].hist(normal_to_tumor, bins=50, alpha=0.5, label='Normal samples', color='blue')
    axes[0].set_xlabel('Cosine Similarity')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Similarity to Tumor Prototype')
    axes[0].legend()
    
    # Similarity to normal prototype
    axes[1].hist(tumor_to_normal, bins=50, alpha=0.5, label='Tumor samples', color='red')
    axes[1].hist(normal_to_normal, bins=50, alpha=0.5, label='Normal samples', color='blue')
    axes[1].set_xlabel('Cosine Similarity')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Similarity to Normal Prototype')
    axes[1].legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    # Demo with random data
    np.random.seed(42)
    
    n_samples = 500
    embed_dim = 3328
    
    # Create embeddings with slight class separation
    tumor_center = np.random.randn(embed_dim)
    normal_center = tumor_center + 0.1 * np.random.randn(embed_dim)
    
    tumor_emb = tumor_center + 0.3 * np.random.randn(n_samples // 2, embed_dim)
    normal_emb = normal_center + 0.3 * np.random.randn(n_samples // 2, embed_dim)
    
    embeddings = np.vstack([tumor_emb, normal_emb]).astype(np.float32)
    labels = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))
    
    # Analyze
    print("Class Separation Analysis:")
    metrics = analyze_class_separation(embeddings, labels)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✓ Embedding analysis module loaded successfully")
