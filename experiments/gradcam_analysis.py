"""
Grad-CAM Analysis: Visualizing Model Attention

This module provides Grad-CAM visualization to understand where the model
attends when making predictions, comparing prototype-based vs text-anchored
inference patterns.

Key insight:
    - Prototype-based: Diffuse attention, species-specific features
    - Text-anchored: Focused on conserved morphological features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class GradCAMAnalyzer:
    """
    Grad-CAM visualization for CPath-CLIP vision encoder.
    
    Generates attention heatmaps showing which regions of the image
    the model focuses on when making predictions.
    
    Args:
        model: CPathOmniInference instance
        target_layer: Which transformer layer to visualize (default: last)
    
    Example:
        >>> analyzer = GradCAMAnalyzer(model)
        >>> heatmap = analyzer.generate_heatmap(image, "tumor")
        >>> analyzer.visualize(image, heatmap, title="Tumor Attention")
    """
    
    def __init__(self, model, target_layer: int = -1):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # Register hooks on vision encoder
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations."""
        visual = self.model.vision_encoder.backbone.visual
        
        # Get target layer (last transformer block by default)
        if self.target_layer == -1:
            target = visual.transformer.resblocks[-1]
        else:
            target = visual.transformer.resblocks[self.target_layer]
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target.register_forward_hook(forward_hook)
        target.register_full_backward_hook(backward_hook)
    
    def generate_heatmap(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        target_class: str = "tumor",
        prompts: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an image.
        
        Args:
            image: Input image
            target_class: Class to visualize ('tumor' or 'normal')
            prompts: Custom prompts (optional)
        
        Returns:
            Heatmap as numpy array (H, W) with values 0-1
        """
        # Preprocess image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        img_tensor = self.model.vision_encoder.preprocess(image).unsqueeze(0)
        img_tensor = img_tensor.to(self.model.device)
        img_tensor.requires_grad_(True)
        
        # Forward pass
        self.model.vision_encoder.backbone.zero_grad()
        
        # Get image embedding
        image_emb = self.model.vision_encoder.backbone.encode_image(img_tensor)
        
        # Get target text embedding
        from models.text_encoder import get_prompts
        if prompts is None:
            prompts = get_prompts("breast_cancer")
        
        text_embs = self.model.get_text_embeddings(prompts)
        target_emb = text_embs[target_class]
        
        # Compute similarity as the target
        similarity = F.cosine_similarity(image_emb, target_emb.unsqueeze(0))
        
        # Backward pass
        similarity.backward()
        
        # Compute Grad-CAM
        if self.activations is not None and self.gradients is not None:
            # Global average pooling of gradients
            weights = self.gradients.mean(dim=0)  # (seq_len, hidden_dim)
            
            # Weighted combination of activations
            cam = (weights * self.activations.squeeze(1)).sum(dim=-1)  # (seq_len,)
            
            # Remove CLS token
            cam = cam[1:]  # (num_patches,)
            
            # Reshape to spatial dimensions
            # For ViT-L/14-336: 336/14 = 24 patches per side
            grid_size = int(np.sqrt(cam.shape[0]))
            cam = cam.reshape(grid_size, grid_size).cpu().numpy()
            
            # Apply ReLU and normalize
            cam = np.maximum(cam, 0)
            if cam.max() > 0:
                cam = cam / cam.max()
            
            # Resize to image size
            cam_resized = np.array(Image.fromarray(cam).resize(
                (image.size[0], image.size[1]), 
                Image.BILINEAR
            ))
            
            return cam_resized
        
        return np.zeros((image.size[1], image.size[0]))
    
    def visualize(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        heatmap: np.ndarray,
        title: str = "",
        alpha: float = 0.5,
        colormap: str = "jet",
        save_path: Optional[str] = None
    ):
        """
        Visualize Grad-CAM heatmap overlaid on image.
        
        Args:
            image: Original image
            heatmap: Grad-CAM heatmap
            title: Plot title
            alpha: Heatmap transparency
            colormap: Matplotlib colormap
            save_path: Path to save figure
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image_np = np.array(image)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)
        plt.imshow(heatmap, cmap=colormap, alpha=alpha)
        plt.title(title, fontsize=14)
        plt.axis('off')
        plt.colorbar(label='Activation')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✓ Saved: {save_path}")
        
        plt.close()


def generate_gradcam_comparison(
    model,
    image: Union[str, Path, Image.Image],
    output_path: str,
    title: str = "Grad-CAM Comparison"
) -> str:
    """
    Generate side-by-side Grad-CAM comparison: Prototype vs Text-Anchored.
    
    Args:
        model: CPathOmniInference instance
        image: Input image
        output_path: Path to save comparison figure
        title: Figure title
    
    Returns:
        Path to saved figure
    """
    # Load image
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')
    
    image_np = np.array(image)
    
    # Initialize analyzer
    analyzer = GradCAMAnalyzer(model)
    
    # Generate heatmaps for both methods
    heatmap_tumor = analyzer.generate_heatmap(image, "tumor")
    heatmap_normal = analyzer.generate_heatmap(image, "normal")
    
    # Create comparison figure
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # Original image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(image_np)
    ax1.set_title("Original", fontsize=12)
    ax1.axis('off')
    
    # Tumor attention
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(image_np)
    im2 = ax2.imshow(heatmap_tumor, cmap='jet', alpha=0.5)
    ax2.set_title("Tumor Attention", fontsize=12)
    ax2.axis('off')
    
    # Normal attention
    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(image_np)
    im3 = ax3.imshow(heatmap_normal, cmap='jet', alpha=0.5)
    ax3.set_title("Normal Attention", fontsize=12)
    ax3.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"✓ Saved Grad-CAM comparison: {output_path}")
    return output_path


def compute_attention_iou(
    heatmap1: np.ndarray,
    heatmap2: np.ndarray,
    threshold_percentile: float = 90
) -> float:
    """
    Compute IoU between top activation regions of two heatmaps.
    
    This metric quantifies how much two attention patterns overlap,
    useful for comparing prototype vs text-anchored attention.
    
    Args:
        heatmap1: First heatmap
        heatmap2: Second heatmap
        threshold_percentile: Percentile for top activation (default: 90%)
    
    Returns:
        IoU score (0-1)
    """
    # Threshold to get top activations
    thresh1 = np.percentile(heatmap1, threshold_percentile)
    thresh2 = np.percentile(heatmap2, threshold_percentile)
    
    mask1 = heatmap1 >= thresh1
    mask2 = heatmap2 >= thresh2
    
    # Compute IoU
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    iou = intersection / union if union > 0 else 0.0
    return iou


def analyze_attention_patterns(
    model,
    images: List[Union[str, Path]],
    output_dir: str
) -> Dict:
    """
    Analyze attention patterns across multiple images.
    
    Computes statistics on attention distribution, focus regions,
    and compares prototype vs text-anchored patterns.
    
    Args:
        model: CPathOmniInference instance
        images: List of image paths
        output_dir: Directory to save analysis results
    
    Returns:
        Dictionary with analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer = GradCAMAnalyzer(model)
    
    results = {
        "ious": [],
        "tumor_focus": [],
        "normal_focus": []
    }
    
    for i, img_path in enumerate(images):
        print(f"Analyzing image {i+1}/{len(images)}: {Path(img_path).name}")
        
        # Generate heatmaps
        heatmap_tumor = analyzer.generate_heatmap(img_path, "tumor")
        heatmap_normal = analyzer.generate_heatmap(img_path, "normal")
        
        # Compute metrics
        iou = compute_attention_iou(heatmap_tumor, heatmap_normal)
        results["ious"].append(iou)
        
        # Focus metric: how concentrated is the attention?
        results["tumor_focus"].append(heatmap_tumor.max() - heatmap_tumor.mean())
        results["normal_focus"].append(heatmap_normal.max() - heatmap_normal.mean())
        
        # Save individual visualizations
        generate_gradcam_comparison(
            model, img_path, 
            str(output_dir / f"gradcam_{i:03d}.png"),
            title=f"Image {i+1}"
        )
    
    # Summary statistics
    summary = {
        "mean_iou": np.mean(results["ious"]),
        "std_iou": np.std(results["ious"]),
        "mean_tumor_focus": np.mean(results["tumor_focus"]),
        "mean_normal_focus": np.mean(results["normal_focus"]),
        "n_images": len(images)
    }
    
    print(f"\n{'='*50}")
    print("Attention Pattern Analysis Summary")
    print(f"{'='*50}")
    print(f"Mean IoU@90%: {summary['mean_iou']:.3f} ± {summary['std_iou']:.3f}")
    print(f"Mean Tumor Focus: {summary['mean_tumor_focus']:.3f}")
    print(f"Mean Normal Focus: {summary['mean_normal_focus']:.3f}")
    
    # Save summary
    import json
    with open(output_dir / "attention_analysis.json", "w") as f:
        json.dump({**results, **summary}, f, indent=2)
    
    return {**results, **summary}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Grad-CAM Analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--image-dir", type=str, help="Directory of images")
    parser.add_argument("--output-dir", type=str, default="gradcam_results")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    from models.inference import CPathOmniInference
    
    model = CPathOmniInference(
        vision_encoder_path=args.checkpoint,
        text_encoder="Qwen/Qwen2-1.5B",
        device=args.device
    )
    
    if args.image:
        generate_gradcam_comparison(
            model, args.image,
            f"{args.output_dir}/gradcam_comparison.png"
        )
    elif args.image_dir:
        images = list(Path(args.image_dir).glob("*.png")) + list(Path(args.image_dir).glob("*.jpg"))
        analyze_attention_patterns(model, images[:20], args.output_dir)
