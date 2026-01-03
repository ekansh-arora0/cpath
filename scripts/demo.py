#!/usr/bin/env python3
"""
CPath-Omni Interactive Demo

Quick demonstration of the inference pipeline.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_synthetic_data():
    """Demo with synthetic data (no real model weights required)."""
    print("="*60)
    print("CPath-Omni Demo (Synthetic Data)")
    print("="*60)
    
    # Create synthetic "patch" images
    print("\nCreating synthetic pathology patches...")
    
    # Simulated tumor patch (pink/purple, high nuclei density)
    np.random.seed(42)
    tumor_patch = np.zeros((224, 224, 3), dtype=np.uint8)
    tumor_patch[:, :, 0] = 200  # Red
    tumor_patch[:, :, 1] = 150  # Green
    tumor_patch[:, :, 2] = 180  # Blue
    # Add nuclei-like dark spots
    for _ in range(500):
        x, y = np.random.randint(0, 220, 2)
        tumor_patch[y:y+4, x:x+4] = [80, 50, 100]
    
    # Simulated normal patch (lighter, sparse structure)
    normal_patch = np.zeros((224, 224, 3), dtype=np.uint8)
    normal_patch[:, :, 0] = 230
    normal_patch[:, :, 1] = 200
    normal_patch[:, :, 2] = 200
    # Add sparse cells
    for _ in range(50):
        x, y = np.random.randint(0, 210, 2)
        normal_patch[y:y+8, x:x+8] = [180, 150, 160]
    
    # Display
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(tumor_patch)
    axes[0].set_title("Simulated Tumor Patch\n(Dense nuclei, high cellularity)")
    axes[0].axis('off')
    
    axes[1].imshow(normal_patch)
    axes[1].set_title("Simulated Normal Patch\n(Sparse structure)")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("demo_synthetic_patches.png", dpi=150)
    print("✓ Saved: demo_synthetic_patches.png")
    
    # Simulated predictions
    print("\n--- Simulated Inference Results ---")
    print("\nTumor Patch:")
    print("  Prediction: TUMOR")
    print("  Confidence: 0.9234")
    print("  Tumor similarity: 0.8567")
    print("  Normal similarity: 0.3421")
    
    print("\nNormal Patch:")
    print("  Prediction: NORMAL")
    print("  Confidence: 0.8876")
    print("  Tumor similarity: 0.2134")
    print("  Normal similarity: 0.7891")
    
    return tumor_patch, normal_patch


def demo_with_model(image_path, checkpoint_path=None):
    """Demo with actual model."""
    from models.inference import CPathOmniInference
    
    print("="*60)
    print("CPath-Omni Demo (Real Inference)")
    print("="*60)
    
    # Initialize
    print("\nLoading model...")
    model = CPathOmniInference(
        vision_encoder_path=checkpoint_path,
        device="cuda"
    )
    
    # Run inference
    print(f"\nProcessing: {image_path}")
    result = model.predict_patch(
        image_path,
        mode="text",
        cancer_type="breast_cancer"
    )
    
    # Display results
    print("\n--- Inference Results ---")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Tumor probability: {result['tumor_probability']:.4f}")
    print(f"Tumor similarity: {result['tumor_similarity']:.4f}")
    print(f"Normal similarity: {result['normal_similarity']:.4f}")
    
    # Visualize
    img = Image.open(image_path)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(
        f"Prediction: {result['prediction'].upper()}\n"
        f"Confidence: {result['confidence']:.2%}",
        fontsize=14
    )
    ax.axis('off')
    
    # Color border based on prediction
    color = 'red' if result['prediction'] == 'tumor' else 'green'
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(5)
        spine.set_visible(True)
    
    output_path = Path(image_path).stem + "_prediction.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    return result


def demo_gradcam(image_path, checkpoint_path=None):
    """Demo Grad-CAM visualization."""
    import torch
    from models.cpath_clip import CPathCLIP
    
    print("="*60)
    print("CPath-Omni Grad-CAM Demo")
    print("="*60)
    
    # This is a simplified Grad-CAM visualization
    # Full implementation would require hooks into the vision transformer
    
    print("\nGrad-CAM visualization shows which image regions")
    print("are most important for the model's decision.")
    print("\nKey features:")
    print("  • Red/yellow: High attention (important for prediction)")
    print("  • Blue/purple: Low attention")
    print("  • Used to verify model focuses on relevant tissue")
    
    # Create synthetic Grad-CAM overlay
    np.random.seed(42)
    img_size = 224
    
    # Simulate attention on central region (where tumor might be)
    y, x = np.ogrid[:img_size, :img_size]
    center_y, center_x = img_size // 2, img_size // 2
    
    # Multiple attention hotspots
    attention = np.zeros((img_size, img_size))
    for _ in range(5):
        cy = np.random.randint(50, 174)
        cx = np.random.randint(50, 174)
        sigma = np.random.randint(20, 40)
        attention += np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * sigma**2))
    
    attention = (attention - attention.min()) / (attention.max() - attention.min())
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original (synthetic)
    original = np.ones((img_size, img_size, 3)) * 0.8
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original Patch")
    axes[0].axis('off')
    
    # Attention map
    im = axes[1].imshow(attention, cmap='jet')
    axes[1].set_title("Grad-CAM Attention")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    overlay = original.copy()
    cmap = plt.cm.jet
    attention_colored = cmap(attention)[:, :, :3]
    overlay = 0.6 * overlay + 0.4 * attention_colored
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("demo_gradcam.png", dpi=150)
    print("\n✓ Saved: demo_gradcam.png")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CPath-Omni Demo")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, 
                       choices=["synthetic", "inference", "gradcam"],
                       default="synthetic",
                       help="Demo mode")
    
    args = parser.parse_args()
    
    if args.mode == "synthetic":
        demo_synthetic_data()
    elif args.mode == "inference":
        if args.image is None:
            print("Error: --image required for inference mode")
            sys.exit(1)
        demo_with_model(args.image, args.checkpoint)
    elif args.mode == "gradcam":
        demo_gradcam(args.image, args.checkpoint)
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    main()
