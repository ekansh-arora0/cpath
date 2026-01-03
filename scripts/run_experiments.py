#!/usr/bin/env python3
"""
Run All CPath-Omni Experiments

Reproduces the main results from the paper.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.exp1_same_cancer import SameCancerExperiment
from experiments.exp2_cross_cancer import CrossCancerExperiment
from experiments.exp3_cross_species import CrossSpeciesExperiment


def run_experiment_1(args, output_dir):
    """Experiment 1: Same Cancer Type Transfer (Dog Breast → Dog Breast)."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Same Cancer Type Transfer")
    print("Dog Breast Cancer → Dog Breast Cancer")
    print("="*60)
    
    exp = SameCancerExperiment(
        data_path=args.data_dir / "dog_breast",
        cache_dir=args.cache_dir,
        vision_encoder_path=args.checkpoint,
        device=args.device
    )
    
    # Run with different prompt families
    results = exp.run(
        train_fraction=0.8,
        n_prototypes=50,
        prompt_families=[
            "professional", "casual", "descriptive",
            "clinical", "morphological"
        ]
    )
    
    # Save results
    output_path = output_dir / "exp1_same_cancer.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n--- Results Summary ---")
    for mode, metrics in results["evaluation"].items():
        if isinstance(metrics, dict) and "auc" in metrics:
            print(f"{mode}: AUC = {metrics['auc']:.4f} ± {metrics.get('auc_std', 0):.4f}")
    
    return results


def run_experiment_2(args, output_dir):
    """Experiment 2: Cross-Cancer Type Transfer (Mast Cell → Dog Breast)."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Cross-Cancer Type Transfer")
    print("Mast Cell Tumor → Dog Breast Cancer")
    print("="*60)
    
    exp = CrossCancerExperiment(
        source_path=args.data_dir / "mast_cell",
        target_path=args.data_dir / "dog_breast",
        cache_dir=args.cache_dir,
        vision_encoder_path=args.checkpoint,
        device=args.device
    )
    
    results = exp.run(
        n_prototypes=50,
        prompt_families=["professional", "clinical", "morphological"]
    )
    
    # Save results
    output_path = output_dir / "exp2_cross_cancer.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n--- Results Summary ---")
    for mode, metrics in results["evaluation"].items():
        if isinstance(metrics, dict) and "auc" in metrics:
            print(f"{mode}: AUC = {metrics['auc']:.4f} ± {metrics.get('auc_std', 0):.4f}")
    
    return results


def run_experiment_3(args, output_dir):
    """Experiment 3: Cross-Species Transfer (Human → Dog)."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Cross-Species Transfer")
    print("Human TCGA-BRCA → Dog Breast Cancer")
    print("="*60)
    
    exp = CrossSpeciesExperiment(
        human_path=args.data_dir / "tcga_brca",
        dog_path=args.data_dir / "dog_breast",
        cache_dir=args.cache_dir,
        vision_encoder_path=args.checkpoint,
        device=args.device
    )
    
    results = exp.run(
        n_prototypes=50,
        prompt_families=["professional", "clinical", "morphological"]
    )
    
    # Save results
    output_path = output_dir / "exp3_cross_species.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n--- Results Summary ---")
    for mode, metrics in results["evaluation"].items():
        if isinstance(metrics, dict) and "auc" in metrics:
            print(f"{mode}: AUC = {metrics['auc']:.4f} ± {metrics.get('auc_std', 0):.4f}")
    
    return results


def run_ablation(args, output_dir):
    """Ablation studies on key components."""
    print("\n" + "="*60)
    print("ABLATION STUDIES")
    print("="*60)
    
    exp = SameCancerExperiment(
        data_path=args.data_dir / "dog_breast",
        cache_dir=args.cache_dir,
        vision_encoder_path=args.checkpoint,
        device=args.device
    )
    
    ablation_results = {}
    
    # Ablation 1: Number of prototypes
    print("\n--- Ablation: Number of Prototypes ---")
    for n_proto in [10, 25, 50, 100, 200]:
        results = exp.run(
            train_fraction=0.8,
            n_prototypes=n_proto,
            prompt_families=["professional"]
        )
        ablation_results[f"n_prototypes_{n_proto}"] = results["evaluation"]
        print(f"n={n_proto}: AUC = {results['evaluation']['prototype']['auc']:.4f}")
    
    # Ablation 2: Prompt families
    print("\n--- Ablation: Prompt Families ---")
    families = ["professional", "casual", "descriptive", "clinical", "morphological"]
    for family in families:
        results = exp.run(
            train_fraction=0.8,
            n_prototypes=50,
            prompt_families=[family]
        )
        ablation_results[f"prompt_{family}"] = results["evaluation"]
        print(f"{family}: AUC = {results['evaluation']['text']['auc']:.4f}")
    
    # Save ablation results
    output_path = output_dir / "ablation_studies.json"
    with open(output_path, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    return ablation_results


def main():
    parser = argparse.ArgumentParser(description="Run CPath-Omni Experiments")
    
    parser.add_argument("--data-dir", type=Path, required=True,
                       help="Path to data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                       help="Output directory for results")
    parser.add_argument("--cache-dir", type=Path, default=None,
                       help="Cache directory for embeddings")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to vision encoder checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    
    # Experiment selection
    parser.add_argument("--exp", type=str, nargs="+",
                       choices=["1", "2", "3", "ablation", "all"],
                       default=["all"],
                       help="Which experiments to run")
    
    args = parser.parse_args()
    
    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup cache directory
    if args.cache_dir is None:
        args.cache_dir = args.output_dir / "cache"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Run selected experiments
    experiments = args.exp
    if "all" in experiments:
        experiments = ["1", "2", "3", "ablation"]
    
    all_results = {}
    
    if "1" in experiments:
        all_results["exp1"] = run_experiment_1(args, args.output_dir)
    
    if "2" in experiments:
        all_results["exp2"] = run_experiment_2(args, args.output_dir)
    
    if "3" in experiments:
        all_results["exp3"] = run_experiment_3(args, args.output_dir)
    
    if "ablation" in experiments:
        all_results["ablation"] = run_ablation(args, args.output_dir)
    
    # Save combined results
    combined_path = args.output_dir / "all_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    main()
