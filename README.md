# Cross-Species Pathology Transfer

Zero-shot cross-species pathology classification using text-anchored visual representations.

**Paper**: *Lost in Translation: How Language Re-Aligns Vision for Cross-Species Pathology*

## Results

| Transfer Setting | Prototype | Text-Anchored |
|-----------------|-----------|---------------|
| Same-Cancer (Dog→Dog) | 64.89% | 78.39% |
| Cross-Cancer (Breast→Mast) | 56.84% | 53.73% |
| Cross-Species (Human→Dog) | 63.96% | **78.39%** |

Text anchoring overcomes the species domain gap that causes prototype methods to fail.

## Installation

```bash
git clone https://github.com/ekansh-arora0/cross-species-pathology.git
cd cross-species-pathology
pip install -r requirements.txt
```

## Quick Start

```python
from models.inference import CPathOmniInference

model = CPathOmniInference(
    vision_encoder_path="checkpoints/cpath_clip.pt",
    device="cuda"
)

# Classify patch
result = model.predict_patch(
    "patch.png",
    prompts={
        "tumor": "Malignant tumor tissue with nuclear atypia",
        "normal": "Normal tissue with regular architecture"
    }
)
```

## Repository Structure

```
├── models/
│   ├── cpath_clip.py          # Vision encoder (ViT-L/14-336)
│   ├── text_encoder.py        # Qwen2-1.5B text encoder
│   ├── inference.py           # Inference pipeline
│   └── semantic_anchoring.py  # Text-anchored classification
├── training/
│   └── train_utils.py         # Linear probe, few-shot, adapters
├── preprocessing/
│   ├── macenko_normalizer.py  # Stain normalization
│   └── patch_extraction.py    # WSI → patches
├── experiments/
│   ├── exp1_same_cancer.py    # Dog breast → Dog breast
│   ├── exp2_cross_cancer.py   # Dog breast → Mast cell
│   ├── exp3_cross_species.py  # Human → Dog (main result)
│   └── gradcam_analysis.py    # Attention visualization
├── scripts/
│   ├── reproduce_table3.py    # Reproduce main results
│   └── run_experiments.py     # Run all experiments
├── data/
│   ├── splits.json            # Train/test slide IDs
│   └── splits/                # Per-experiment split files
└── checkpoints/
    └── README.md              # Model weight instructions
```

## Data Splits

All splits in `data/splits/`:

| Dataset | Train | Test | Source |
|---------|-------|------|--------|
| CATCH Canine Breast | 1 slide (2,048 patches) | 20 slides (20,191 patches) | [GitHub](https://github.com/DeepPathology/CATCH) |
| MITOS Mast Cell | — | 7 slides (5,530 patches) | [MITOS](https://mitos-atypia-14.grand-challenge.org/) |
| TCGA-BRCA Human | 1,098 slides (245k patches) | — | [GDC](https://portal.gdc.cancer.gov/) |

## Reproduce Table 3

```bash
# With pre-computed embeddings
python scripts/reproduce_table3.py --embeddings-dir data/embeddings

# From scratch
python scripts/reproduce_table3.py --data-dir data/ --checkpoint checkpoints/cpath_clip.pt
```

## Semantic Anchoring

Core method: project text descriptions into visual embedding space for domain-invariant classification.

```python
from models.semantic_anchoring import SemanticAnchoring

model = SemanticAnchoring(text_encoder, projection_head)
model.set_anchors({
    "tumor": "Malignant tissue with nuclear pleomorphism",
    "normal": "Normal tissue architecture"
})
predictions = model.classify(visual_embeddings)
```

## Requirements

- Python 3.10+
- PyTorch 2.1.0
- transformers 4.36.0
- open-clip-torch 2.23.0

See `requirements.txt` for full list.

## Citation

```bibtex
@article{crossspecies2026,
  title={Lost in Translation: How Language Re-Aligns Vision for Cross-Species Pathology},
  year={2026}
}
```

## License

MIT
