"""
Experiment 4: Grad-CAM Analysis

Generate Grad-CAM visualizations to interpret model predictions
and validate that the model focuses on relevant tissue regions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.cpath_clip import CPathCLIP


@dataclass
class GradCAMResult:
    """Container for Grad-CAM results."""
    image: np.ndarray
    attention_map: np.ndarray
    overlay: np.ndarray
    prediction: str
    confidence: float


class GradCAMAnalysis:
    """
    Grad-CAM analysis for CPath-CLIP vision encoder.
    
    Generates visual explanations showing which image regions
    influenced the model's classification decision.
    """
    
    def __init__(
        self,
        vision_encoder_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize Grad-CAM analyzer.
        
        Args:
            vision_encoder_path: Path to vision encoder checkpoint
            device: Computation device
        """
        self.device = device
        
        # Load vision encoder
        self.model = CPathCLIP(checkpoint_path=vision_encoder_path)
        self.model.model.to(device)
        self.model.model.eval()
        
        # For storing gradients
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for Grad-CAM."""
        # Get the last transformer block
        visual = self.model.model.visual
        
        # For ViT, we hook into the last attention block
        if hasattr(visual, 'transformer'):
            target_layer = visual.transformer.resblocks[-1]
        elif hasattr(visual, 'blocks'):
            target_layer = visual.blocks[-1]
        else:
            raise ValueError("Cannot find transformer blocks in model")
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def compute_gradcam(
        self,
        image: Image.Image,
        target_text: str = "tumor"
    ) -> np.ndarray:
        """
        Compute Grad-CAM attention map for an image.
        
        Args:
            image: Input PIL Image
            target_text: Text to compute gradients against
            
        Returns:
            attention_map: 2D attention map (same size as image patches)
        """
        # Preprocess image
        from models.cpath_clip import CPathCLIP
        processed = self.model.preprocess(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        self.model.model.zero_grad()
        image_features = self.model.model.encode_image(processed)
        image_features = F.normalize(image_features, dim=-1)
        
        # Get text features
        text_features = self.model.get_text_embedding(target_text)
        text_features = torch.tensor(text_features).to(self.device)
        text_features = F.normalize(text_features.unsqueeze(0), dim=-1)
        
        # Compute similarity and backprop
        similarity = (image_features @ text_features.T).squeeze()
        similarity.backward()
        
        # Compute Grad-CAM
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients not captured. Check hook registration.")
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=1, keepdim=True)  # [B, 1, D]
        
        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=-1)  # [B, N]
        
        # Remove CLS token if present
        if cam.shape[1] == 577:  # 24*24 + 1 for ViT-L/14-336
            cam = cam[:, 1:]
        
        # Reshape to spatial dimensions
        h = w = int(np.sqrt(cam.shape[1]))
        cam = cam.view(1, h, w)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Resize to image size
        cam = F.interpolate(
            cam.unsqueeze(0),
            size=(image.size[1], image.size[0]),
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        return cam
    
    def create_overlay(
        self,
        image: np.ndarray,
        attention_map: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Create attention overlay on image.
        
        Args:
            image: Original image as numpy array
            attention_map: 2D attention map
            alpha: Overlay transparency
            colormap: Matplotlib colormap name
            
        Returns:
            overlay: Image with attention overlay
        """
        # Ensure image is float
        if image.max() > 1:
            image = image / 255.0
        
        # Apply colormap to attention
        cmap = plt.cm.get_cmap(colormap)
        attention_colored = cmap(attention_map)[:, :, :3]
        
        # Blend
        overlay = (1 - alpha) * image + alpha * attention_colored
        overlay = np.clip(overlay, 0, 1)
        
        return (overlay * 255).astype(np.uint8)
    
    def analyze_patch(
        self,
        image_path: str,
        tumor_prompt: str = "tumor tissue",
        normal_prompt: str = "normal tissue"
    ) -> GradCAMResult:
        """
        Full Grad-CAM analysis on a single patch.
        
        Args:
            image_path: Path to image file
            tumor_prompt: Text prompt for tumor class
            normal_prompt: Text prompt for normal class
            
        Returns:
            GradCAMResult with image, attention, overlay, and prediction
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Get prediction
        processed = self.model.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.model.encode_image(processed)
            image_features = F.normalize(image_features, dim=-1)
        
        # Get text embeddings
        tumor_emb = torch.tensor(
            self.model.get_text_embedding(tumor_prompt)
        ).to(self.device)
        normal_emb = torch.tensor(
            self.model.get_text_embedding(normal_prompt)
        ).to(self.device)
        
        tumor_emb = F.normalize(tumor_emb.unsqueeze(0), dim=-1)
        normal_emb = F.normalize(normal_emb.unsqueeze(0), dim=-1)
        
        # Compute similarities
        tumor_sim = (image_features @ tumor_emb.T).item()
        normal_sim = (image_features @ normal_emb.T).item()
        
        # Prediction
        if tumor_sim > normal_sim:
            prediction = "tumor"
            confidence = torch.softmax(
                torch.tensor([tumor_sim, normal_sim]) * 100, dim=0
            )[0].item()
            target_prompt = tumor_prompt
        else:
            prediction = "normal"
            confidence = torch.softmax(
                torch.tensor([tumor_sim, normal_sim]) * 100, dim=0
            )[1].item()
            target_prompt = normal_prompt
        
        # Compute Grad-CAM for predicted class
        attention_map = self.compute_gradcam(image, target_prompt)
        
        # Create overlay
        overlay = self.create_overlay(image_np, attention_map)
        
        return GradCAMResult(
            image=image_np,
            attention_map=attention_map,
            overlay=overlay,
            prediction=prediction,
            confidence=confidence
        )
    
    def compute_iou_at_threshold(
        self,
        attention_map: np.ndarray,
        ground_truth_mask: np.ndarray,
        threshold: float = 0.1
    ) -> float:
        """
        Compute IoU between top attention regions and ground truth.
        
        Args:
            attention_map: Normalized attention map [0, 1]
            ground_truth_mask: Binary mask of tumor region
            threshold: Top percentage of attention to consider
            
        Returns:
            iou: Intersection over Union score
        """
        # Threshold attention (top X%)
        attention_threshold = np.percentile(attention_map, 100 * (1 - threshold))
        attention_binary = attention_map >= attention_threshold
        
        # Compute IoU
        intersection = np.logical_and(attention_binary, ground_truth_mask).sum()
        union = np.logical_or(attention_binary, ground_truth_mask).sum()
        
        iou = intersection / (union + 1e-8)
        return iou
    
    def create_comparison_figure(
        self,
        results: List[GradCAMResult],
        titles: List[str],
        output_path: str,
        figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Create comparison figure with multiple Grad-CAM results.
        
        Args:
            results: List of GradCAMResult objects
            titles: Titles for each result
            output_path: Path to save figure
            figsize: Figure size
        """
        n = len(results)
        fig, axes = plt.subplots(n, 3, figsize=figsize)
        
        if n == 1:
            axes = axes.reshape(1, -1)
        
        for i, (result, title) in enumerate(zip(results, titles)):
            # Original
            axes[i, 0].imshow(result.image)
            axes[i, 0].set_title(f"{title}\nOriginal")
            axes[i, 0].axis('off')
            
            # Attention map
            im = axes[i, 1].imshow(result.attention_map, cmap='jet')
            axes[i, 1].set_title("Attention Map")
            axes[i, 1].axis('off')
            plt.colorbar(im, ax=axes[i, 1], fraction=0.046)
            
            # Overlay
            axes[i, 2].imshow(result.overlay)
            axes[i, 2].set_title(
                f"Overlay\nPred: {result.prediction.upper()} "
                f"({result.confidence:.1%})"
            )
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_path}")


def run_gradcam_experiment(
    data_dir: str,
    output_dir: str,
    vision_encoder_path: Optional[str] = None,
    device: str = "cuda"
) -> Dict:
    """
    Run full Grad-CAM analysis experiment.
    
    Args:
        data_dir: Directory containing test images
        output_dir: Output directory for visualizations
        vision_encoder_path: Path to vision encoder checkpoint
        device: Computation device
        
    Returns:
        Dictionary with analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = GradCAMAnalysis(
        vision_encoder_path=vision_encoder_path,
        device=device
    )
    
    # Find test images
    data_path = Path(data_dir)
    tumor_images = list((data_path / "tumor").glob("*.png"))[:5]
    normal_images = list((data_path / "normal").glob("*.png"))[:5]
    
    results = {
        "tumor": [],
        "normal": []
    }
    
    # Analyze tumor images
    print("Analyzing tumor images...")
    tumor_results = []
    for img_path in tumor_images:
        result = analyzer.analyze_patch(str(img_path))
        tumor_results.append(result)
        results["tumor"].append({
            "image": str(img_path),
            "prediction": result.prediction,
            "confidence": result.confidence,
            "correct": result.prediction == "tumor"
        })
    
    # Analyze normal images
    print("Analyzing normal images...")
    normal_results = []
    for img_path in normal_images:
        result = analyzer.analyze_patch(str(img_path))
        normal_results.append(result)
        results["normal"].append({
            "image": str(img_path),
            "prediction": result.prediction,
            "confidence": result.confidence,
            "correct": result.prediction == "normal"
        })
    
    # Create comparison figures
    if tumor_results:
        analyzer.create_comparison_figure(
            tumor_results,
            [f"Tumor {i+1}" for i in range(len(tumor_results))],
            str(output_dir / "gradcam_tumor_samples.png")
        )
    
    if normal_results:
        analyzer.create_comparison_figure(
            normal_results,
            [f"Normal {i+1}" for i in range(len(normal_results))],
            str(output_dir / "gradcam_normal_samples.png")
        )
    
    # Compute summary statistics
    tumor_accuracy = sum(r["correct"] for r in results["tumor"]) / len(results["tumor"]) if results["tumor"] else 0
    normal_accuracy = sum(r["correct"] for r in results["normal"]) / len(results["normal"]) if results["normal"] else 0
    
    results["summary"] = {
        "tumor_accuracy": tumor_accuracy,
        "normal_accuracy": normal_accuracy,
        "overall_accuracy": (tumor_accuracy + normal_accuracy) / 2
    }
    
    print(f"\n--- Results ---")
    print(f"Tumor accuracy: {tumor_accuracy:.1%}")
    print(f"Normal accuracy: {normal_accuracy:.1%}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Grad-CAM Analysis")
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Path to test data")
    parser.add_argument("--output-dir", type=str, default="results/gradcam",
                       help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Vision encoder checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")
    
    args = parser.parse_args()
    
    run_gradcam_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vision_encoder_path=args.checkpoint,
        device=args.device
    )
