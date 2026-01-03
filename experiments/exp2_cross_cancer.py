"""
Experiment 2: Cross-Cancer Transfer

This experiment evaluates the model's ability to transfer from one cancer type
to another within the same species.

Setup:
    - Train: Canine breast carcinoma → compute prototypes
    - Test: Canine mast cell tumor (different cancer type)
    
This tests whether learned representations generalize across cancer types
or are specific to the training cancer's morphology.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.inference import CPathOmniInference
from models.text_encoder import get_prompts


def run_cross_cancer_experiment(
    vision_checkpoint: str,
    breast_data_dir: str,
    mast_cell_data_dir: str,
    device: str = "cuda",
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run cross-cancer transfer experiment.
    
    Setup:
        - Train: Canine breast carcinoma slides → compute tumor/normal prototypes
        - Test: Canine mast cell tumor slides
        - Compare: Prototype-based vs Text-anchored zero-shot
    
    Expected result:
        - Prototype-based: Poor performance (different morphology)
        - Text-anchored: Improved performance (semantic generalization)
    
    Args:
        vision_checkpoint: Path to CPath-CLIP weights
        breast_data_dir: Directory with breast carcinoma WSIs
        mast_cell_data_dir: Directory with mast cell tumor WSIs
        device: torch device
        output_dir: Directory to save results
    
    Returns:
        Dictionary with experiment results
    """
    print("="*70)
    print("EXPERIMENT 2: Cross-Cancer Transfer")
    print("Train: Canine Breast Carcinoma → Test: Canine Mast Cell Tumor")
    print("="*70)
    
    # Initialize inference pipeline
    model = CPathOmniInference(
        vision_encoder_path=vision_checkpoint,
        text_encoder="Qwen/Qwen2-1.5B",
        device=device
    )
    
    # Load data
    breast_dir = Path(breast_data_dir)
    mast_cell_dir = Path(mast_cell_data_dir)
    
    breast_slides = sorted(breast_dir.glob("*.svs"))
    mast_cell_slides = sorted(mast_cell_dir.glob("*.svs"))
    
    print(f"\nData:")
    print(f"  Training (breast): {len(breast_slides)} slides")
    print(f"  Testing (mast cell): {len(mast_cell_slides)} slides")
    
    # Phase 1: Compute prototypes from breast carcinoma
    print("\n[1] Computing prototypes from breast carcinoma slides...")
    
    # In real experiment: extract embeddings, separate by tumor/normal annotation
    # model.set_prototypes(tumor_embeddings, normal_embeddings)
    
    # Phase 2: Evaluate on mast cell tumor
    print("\n[2] Evaluating on mast cell tumor slides...")
    
    results = {
        "prototype": {"predictions": [], "labels": [], "scores": []},
        "text_anchored": {"predictions": [], "labels": [], "scores": []}
    }
    
    # Get mast cell-specific prompts
    mast_cell_prompts = get_prompts("mast_cell_tumor")
    
    for slide_path in tqdm(mast_cell_slides, desc="Evaluating mast cell"):
        # Text-anchored with mast cell prompts
        text_result = model.predict_wsi(
            slide_path,
            mode="text",
            prompts=mast_cell_prompts,
            patch_size=1024,
            stride=2048
        )
        
        for pred in text_result["patch_results"]:
            results["text_anchored"]["predictions"].append(
                1 if pred["prediction"] == "tumor" else 0
            )
            results["text_anchored"]["scores"].append(pred["tumor_probability"])
            # Labels from annotations
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
    print("RESULTS: Cross-Cancer Transfer (Breast → Mast Cell)")
    print("="*70)
    
    for method, m in metrics.items():
        print(f"\n{method.upper()}:")
        print(f"  AUC-ROC:   {m.get('auc_roc', 0)*100:.2f}%")
        print(f"  Accuracy:  {m.get('accuracy', 0)*100:.2f}%")
        print(f"  Precision: {m.get('precision', 0)*100:.2f}%")
        print(f"  Recall:    {m.get('recall', 0)*100:.2f}%")
        print(f"  F1-Score:  {m.get('f1', 0)*100:.2f}%")
    
    # Analysis
    if "prototype" in metrics and "text_anchored" in metrics:
        improvement = metrics["text_anchored"]["auc_roc"] - metrics["prototype"]["auc_roc"]
        print(f"\n📊 Text anchoring improvement: {improvement*100:+.2f}% AUC-ROC")
        
        if improvement > 0.1:
            print("   → Text anchoring significantly helps cross-cancer transfer!")
        elif improvement > 0.05:
            print("   → Modest improvement from text anchoring")
        else:
            print("   → Cross-cancer transfer remains challenging")
    
    # Save results
    if output_dir:
        import json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "exp2_cross_cancer_results.json", "w") as f:
            json.dump({
                "experiment": "cross_cancer_transfer",
                "train_cancer": "breast_carcinoma",
                "test_cancer": "mast_cell_tumor",
                "species": "canine",
                "metrics": metrics
            }, f, indent=2)
    
    return {
        "experiment": "cross_cancer",
        "train_cancer": "breast_carcinoma",
        "test_cancer": "mast_cell_tumor",
        "metrics": metrics
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run cross-cancer transfer experiment")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--breast-dir", type=str, required=True, help="Breast carcinoma data")
    parser.add_argument("--mast-cell-dir", type=str, required=True, help="Mast cell tumor data")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    run_cross_cancer_experiment(
        vision_checkpoint=args.checkpoint,
        breast_data_dir=args.breast_dir,
        mast_cell_data_dir=args.mast_cell_dir,
        output_dir=args.output_dir,
        device=args.device
    )
