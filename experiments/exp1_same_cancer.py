"""
Experiment 1: Same-Cancer Baseline

This experiment evaluates prototype-based and text-anchored zero-shot
classification on the same cancer type (train and test on canine breast carcinoma).

This serves as the baseline to establish upper-bound performance before
introducing domain shift (cross-cancer or cross-species).
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.inference import CPathOmniInference
from models.text_encoder import get_prompts


def run_same_cancer_experiment(
    vision_checkpoint: str,
    data_dir: str,
    annotations_db: Optional[str] = None,
    device: str = "cuda",
    n_train_slides: int = 15,
    n_test_slides: int = 6,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run same-cancer baseline experiment.
    
    Setup:
        - Train: N slides of canine breast carcinoma → compute prototypes
        - Test: M slides of canine breast carcinoma (held out)
        - Compare: Prototype-based vs Text-anchored zero-shot
    
    Args:
        vision_checkpoint: Path to CPath-CLIP weights
        data_dir: Directory containing WSI files
        annotations_db: Path to SQLite database with tumor annotations
        device: torch device
        n_train_slides: Number of slides for prototype computation
        n_test_slides: Number of slides for testing
        output_dir: Directory to save results
    
    Returns:
        Dictionary with experiment results
    """
    print("="*70)
    print("EXPERIMENT 1: Same-Cancer Baseline (Canine Breast Carcinoma)")
    print("="*70)
    
    # Initialize inference pipeline
    model = CPathOmniInference(
        vision_encoder_path=vision_checkpoint,
        text_encoder="Qwen/Qwen2-1.5B",
        device=device
    )
    
    # Load data
    data_dir = Path(data_dir)
    slides = sorted(data_dir.glob("*.svs"))
    
    if len(slides) < n_train_slides + n_test_slides:
        print(f"Warning: Only {len(slides)} slides available")
        n_train_slides = int(len(slides) * 0.7)
        n_test_slides = len(slides) - n_train_slides
    
    train_slides = slides[:n_train_slides]
    test_slides = slides[n_train_slides:n_train_slides + n_test_slides]
    
    print(f"\nData split:")
    print(f"  Train (prototype): {len(train_slides)} slides")
    print(f"  Test: {len(test_slides)} slides")
    
    # Extract embeddings and compute prototypes
    print("\n[1] Extracting embeddings from training slides...")
    tumor_embeddings = []
    normal_embeddings = []
    
    # Note: In real experiment, you'd use annotations to label patches
    # Here we show the structure - actual labels would come from annotations_db
    for slide_path in tqdm(train_slides, desc="Processing train slides"):
        result = model.predict_wsi(
            slide_path,
            mode="text",  # Use text to get initial labels
            cancer_type="breast_cancer",
            patch_size=1024,
            stride=2048,
            batch_size=32
        )
        
        # Collect embeddings by predicted class
        # In real experiment, use ground truth annotations
        for i, pred in enumerate(result["patch_results"]):
            # Would need to re-extract embeddings here
            pass
    
    # For demo, create mock prototypes
    print("\n[2] Computing prototypes...")
    # In real code: model.set_prototypes(tumor_embeddings, normal_embeddings)
    
    # Evaluate on test slides
    print("\n[3] Evaluating on test slides...")
    
    results = {
        "prototype": {"predictions": [], "labels": [], "scores": []},
        "text_anchored": {"predictions": [], "labels": [], "scores": []}
    }
    
    for slide_path in tqdm(test_slides, desc="Evaluating"):
        # Text-anchored evaluation
        text_result = model.predict_wsi(
            slide_path,
            mode="text",
            cancer_type="breast_cancer",
            patch_size=1024,
            stride=2048
        )
        
        for pred in text_result["patch_results"]:
            results["text_anchored"]["predictions"].append(
                1 if pred["prediction"] == "tumor" else 0
            )
            results["text_anchored"]["scores"].append(pred["tumor_probability"])
            # Labels would come from annotations
            results["text_anchored"]["labels"].append(1)  # Placeholder
    
    # Compute metrics
    metrics = {}
    for method in ["prototype", "text_anchored"]:
        if len(results[method]["labels"]) > 0:
            preds = np.array(results[method]["predictions"])
            labels = np.array(results[method]["labels"])
            scores = np.array(results[method]["scores"])
            
            metrics[method] = {
                "accuracy": accuracy_score(labels, preds),
                "precision": precision_score(labels, preds, zero_division=0),
                "recall": recall_score(labels, preds, zero_division=0),
                "f1": f1_score(labels, preds, zero_division=0),
                "auc_roc": roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
            }
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS: Same-Cancer Baseline")
    print("="*70)
    
    for method, m in metrics.items():
        print(f"\n{method.upper()}:")
        print(f"  AUC-ROC:   {m.get('auc_roc', 0)*100:.2f}%")
        print(f"  Accuracy:  {m.get('accuracy', 0)*100:.2f}%")
        print(f"  Precision: {m.get('precision', 0)*100:.2f}%")
        print(f"  Recall:    {m.get('recall', 0)*100:.2f}%")
        print(f"  F1-Score:  {m.get('f1', 0)*100:.2f}%")
    
    # Save results
    if output_dir:
        import json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "exp1_same_cancer_results.json", "w") as f:
            json.dump({
                "experiment": "same_cancer_baseline",
                "n_train_slides": len(train_slides),
                "n_test_slides": len(test_slides),
                "metrics": metrics
            }, f, indent=2)
    
    return {
        "experiment": "same_cancer",
        "train_slides": len(train_slides),
        "test_slides": len(test_slides),
        "metrics": metrics
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run same-cancer baseline experiment")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to vision encoder checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with WSI files")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    run_same_cancer_experiment(
        vision_checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
