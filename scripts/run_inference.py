#!/usr/bin/env python3
"""
Run Inference with CPath-Omni

Command-line interface for running inference on pathology images.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.inference import CPathOmniInference
from models.text_encoder import get_prompts


def main():
    parser = argparse.ArgumentParser(
        description="CPath-Omni Inference Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single patch inference
  python run_inference.py --image patch.png --checkpoint model.pt
  
  # WSI inference
  python run_inference.py --wsi slide.svs --checkpoint model.pt --output results.json
  
  # Batch inference
  python run_inference.py --image-dir patches/ --checkpoint model.pt
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to single image")
    input_group.add_argument("--wsi", type=str, help="Path to whole slide image")
    input_group.add_argument("--image-dir", type=str, help="Directory of images")
    
    # Model options
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to vision encoder checkpoint")
    parser.add_argument("--text-encoder", type=str, default="Qwen/Qwen2-1.5B",
                       help="HuggingFace text encoder model")
    parser.add_argument("--projection", type=str, default=None,
                       help="Path to text projection weights")
    
    # Inference options
    parser.add_argument("--mode", type=str, choices=["text", "prototype"], default="text",
                       help="Inference mode")
    parser.add_argument("--cancer-type", type=str, default="breast_cancer",
                       choices=["breast_cancer", "mast_cell_tumor", "generic"],
                       help="Cancer type for default prompts")
    parser.add_argument("--tumor-prompt", type=str, default=None,
                       help="Custom tumor prompt")
    parser.add_argument("--normal-prompt", type=str, default=None,
                       help="Custom normal prompt")
    
    # WSI options
    parser.add_argument("--patch-size", type=int, default=1024,
                       help="Patch size for WSI processing")
    parser.add_argument("--stride", type=int, default=512,
                       help="Stride for patch extraction")
    parser.add_argument("--tissue-threshold", type=float, default=0.1,
                       help="Minimum tissue fraction")
    parser.add_argument("--no-stain-normalize", action="store_true",
                       help="Disable stain normalization")
    
    # Other options
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path (JSON)")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Initialize model
    print("Initializing CPath-Omni...")
    model = CPathOmniInference(
        vision_encoder_path=args.checkpoint,
        text_encoder=args.text_encoder,
        projection_path=args.projection,
        device=args.device
    )
    
    # Custom prompts
    prompts = None
    if args.tumor_prompt or args.normal_prompt:
        default_prompts = get_prompts(args.cancer_type)
        prompts = {
            "tumor": args.tumor_prompt or default_prompts["tumor"],
            "normal": args.normal_prompt or default_prompts["normal"]
        }
    
    # Run inference
    if args.image:
        # Single image
        print(f"\nProcessing: {args.image}")
        result = model.predict_patch(
            args.image,
            mode=args.mode,
            prompts=prompts,
            cancer_type=args.cancer_type
        )
        
        print(f"\nPrediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Tumor probability: {result['tumor_probability']:.4f}")
        print(f"Tumor similarity: {result['tumor_similarity']:.4f}")
        print(f"Normal similarity: {result['normal_similarity']:.4f}")
        
        results = {"image": args.image, **result}
        
    elif args.wsi:
        # Whole slide image
        print(f"\nProcessing WSI: {args.wsi}")
        result = model.predict_wsi(
            args.wsi,
            mode=args.mode,
            prompts=prompts,
            cancer_type=args.cancer_type,
            patch_size=args.patch_size,
            stride=args.stride,
            tissue_threshold=args.tissue_threshold,
            batch_size=args.batch_size,
            stain_normalize=not args.no_stain_normalize
        )
        
        print(f"\nSlide prediction: {result['slide_prediction'].upper()}")
        print(f"Total patches: {result['total_patches']}")
        print(f"Tumor patches: {result['tumor_patches']} ({result['tumor_fraction']*100:.1f}%)")
        print(f"Normal patches: {result['normal_patches']}")
        print(f"Average tumor probability: {result['avg_tumor_probability']:.4f}")
        
        # Remove patch-level results for compact output
        results = {k: v for k, v in result.items() if k != "patch_results"}
        results["n_patches_processed"] = len(result["patch_results"])
        
    elif args.image_dir:
        # Batch of images
        image_dir = Path(args.image_dir)
        images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        
        print(f"\nProcessing {len(images)} images from {args.image_dir}")
        
        batch_results = model.predict_batch(
            images,
            mode=args.mode,
            prompts=prompts,
            cancer_type=args.cancer_type,
            batch_size=args.batch_size
        )
        
        # Summary
        tumor_count = sum(1 for r in batch_results if r["prediction"] == "tumor")
        
        print(f"\nResults:")
        print(f"  Total images: {len(images)}")
        print(f"  Tumor predictions: {tumor_count}")
        print(f"  Normal predictions: {len(images) - tumor_count}")
        
        results = {
            "image_dir": str(image_dir),
            "total_images": len(images),
            "tumor_predictions": tumor_count,
            "normal_predictions": len(images) - tumor_count,
            "predictions": [
                {"image": str(img), **res}
                for img, res in zip(images, batch_results)
            ]
        }
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
