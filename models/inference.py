"""
CPath-Omni Inference Pipeline

Unified inference pipeline supporting both prototype-based and text-anchored
zero-shot classification for pathology images.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from .cpath_clip import CPathCLIPVisionEncoder
from .text_encoder import QwenTextEncoder, get_prompts


class CPathOmniInference:
    """
    CPath-Omni Inference Pipeline
    
    Supports two modes:
    1. Prototype-based: Compare image embeddings to class prototypes (mean embeddings)
    2. Text-anchored: Compare image embeddings to text prompt embeddings
    
    Args:
        vision_encoder_path: Path to CPath-CLIP checkpoint
        text_encoder: HuggingFace model name or None to disable text mode
        projection_path: Path to pretrained text projection weights
        device: torch device
    
    Example:
        >>> model = CPathOmniInference("checkpoints/cpath_clip.pt")
        >>> 
        >>> # Text-anchored prediction
        >>> result = model.predict_patch("patch.png", mode="text")
        >>> print(result["prediction"], result["confidence"])
        >>>
        >>> # Prototype-based prediction (requires setting prototypes first)
        >>> model.set_prototypes(tumor_embeddings, normal_embeddings)
        >>> result = model.predict_patch("patch.png", mode="prototype")
    """
    
    def __init__(
        self,
        vision_encoder_path: Optional[str] = None,
        text_encoder: str = "Qwen/Qwen2-1.5B",
        projection_path: Optional[str] = None,
        device: str = "cuda"
    ):
        self.device = torch.device(device)
        
        # Load vision encoder
        print("Initializing CPath-Omni inference pipeline...")
        self.vision_encoder = CPathCLIPVisionEncoder(
            checkpoint_path=vision_encoder_path,
            device=device,
            freeze=True
        )
        self.embed_dim = self.vision_encoder.embed_dim
        
        # Load text encoder (optional)
        self.text_encoder = None
        if text_encoder is not None:
            self.text_encoder = QwenTextEncoder(
                model_name=text_encoder,
                projection_path=projection_path,
                device=device,
                vision_dim=self.embed_dim
            )
        
        # Prototypes (for prototype-based inference)
        self.tumor_prototype = None
        self.normal_prototype = None
        
        # Text embeddings cache
        self._text_embeddings_cache = {}
        
        print("✓ CPath-Omni ready!")
    
    def set_prototypes(
        self,
        tumor_embeddings: torch.Tensor,
        normal_embeddings: torch.Tensor
    ):
        """
        Set class prototypes for prototype-based inference.
        
        Args:
            tumor_embeddings: Tensor of tumor patch embeddings (N, D)
            normal_embeddings: Tensor of normal patch embeddings (M, D)
        """
        # Compute mean prototypes
        self.tumor_prototype = tumor_embeddings.mean(dim=0)
        self.tumor_prototype = self.tumor_prototype / self.tumor_prototype.norm()
        
        self.normal_prototype = normal_embeddings.mean(dim=0)
        self.normal_prototype = self.normal_prototype / self.normal_prototype.norm()
        
        print(f"✓ Set prototypes: tumor ({len(tumor_embeddings)} samples), normal ({len(normal_embeddings)} samples)")
    
    def get_text_embeddings(
        self,
        prompts: Optional[Dict[str, str]] = None,
        cancer_type: str = "generic"
    ) -> Dict[str, torch.Tensor]:
        """
        Get text embeddings for prompts.
        
        Args:
            prompts: Dictionary with 'tumor' and 'normal' prompts, or None to use defaults
            cancer_type: Cancer type for default prompts
        
        Returns:
            Dictionary with 'tumor' and 'normal' text embeddings
        """
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not initialized. Pass text_encoder to constructor.")
        
        if prompts is None:
            prompts = get_prompts(cancer_type)
        
        # Check cache
        cache_key = (prompts["tumor"], prompts["normal"])
        if cache_key in self._text_embeddings_cache:
            return self._text_embeddings_cache[cache_key]
        
        # Encode prompts
        tumor_emb = self.text_encoder.encode_text(prompts["tumor"], normalize=True)
        normal_emb = self.text_encoder.encode_text(prompts["normal"], normalize=True)
        
        result = {
            "tumor": tumor_emb.squeeze(0),
            "normal": normal_emb.squeeze(0)
        }
        
        self._text_embeddings_cache[cache_key] = result
        return result
    
    def predict_patch(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        mode: str = "text",
        prompts: Optional[Dict[str, str]] = None,
        cancer_type: str = "generic",
        return_embeddings: bool = False
    ) -> Dict:
        """
        Predict tumor/normal for a single patch.
        
        Args:
            image: Image path, PIL Image, numpy array, or tensor
            mode: 'text' for text-anchored, 'prototype' for prototype-based
            prompts: Custom prompts (for text mode)
            cancer_type: Cancer type for default prompts
            return_embeddings: Whether to return raw embeddings
        
        Returns:
            Dictionary with prediction, confidence, and optionally embeddings
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(image, Image.Image):
            image = self.vision_encoder.preprocess(image).unsqueeze(0)
        
        image = image.to(self.device)
        
        # Get image embedding
        with torch.no_grad():
            image_emb = self.vision_encoder(image, normalize=True).squeeze(0)
        
        # Compute similarities
        if mode == "text":
            text_embs = self.get_text_embeddings(prompts, cancer_type)
            tumor_sim = F.cosine_similarity(image_emb.unsqueeze(0), text_embs["tumor"].unsqueeze(0)).item()
            normal_sim = F.cosine_similarity(image_emb.unsqueeze(0), text_embs["normal"].unsqueeze(0)).item()
        
        elif mode == "prototype":
            if self.tumor_prototype is None or self.normal_prototype is None:
                raise RuntimeError("Prototypes not set. Call set_prototypes() first.")
            
            tumor_sim = F.cosine_similarity(
                image_emb.unsqueeze(0), 
                self.tumor_prototype.unsqueeze(0).to(self.device)
            ).item()
            normal_sim = F.cosine_similarity(
                image_emb.unsqueeze(0),
                self.normal_prototype.unsqueeze(0).to(self.device)
            ).item()
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'text' or 'prototype'.")
        
        # Determine prediction
        prediction = "tumor" if tumor_sim > normal_sim else "normal"
        confidence = max(tumor_sim, normal_sim)
        
        # Compute probability via softmax
        logits = torch.tensor([tumor_sim, normal_sim])
        probs = F.softmax(logits * 10, dim=0)  # Temperature scaling
        tumor_prob = probs[0].item()
        
        result = {
            "prediction": prediction,
            "confidence": confidence,
            "tumor_similarity": tumor_sim,
            "normal_similarity": normal_sim,
            "tumor_probability": tumor_prob,
            "mode": mode
        }
        
        if return_embeddings:
            result["embedding"] = image_emb.cpu()
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        mode: str = "text",
        prompts: Optional[Dict[str, str]] = None,
        cancer_type: str = "generic",
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Predict tumor/normal for a batch of patches.
        
        Args:
            images: List of images
            mode: 'text' or 'prototype'
            prompts: Custom prompts
            cancer_type: Cancer type for default prompts
            batch_size: Batch size for processing
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in tqdm(range(0, len(images), batch_size), desc="Predicting"):
            batch = images[i:i + batch_size]
            for img in batch:
                result = self.predict_patch(
                    img, mode=mode, prompts=prompts, cancer_type=cancer_type
                )
                results.append(result)
        
        return results
    
    def predict_wsi(
        self,
        wsi_path: Union[str, Path],
        mode: str = "text",
        prompts: Optional[Dict[str, str]] = None,
        cancer_type: str = "generic",
        patch_size: int = 1024,
        stride: int = 512,
        tissue_threshold: float = 0.1,
        batch_size: int = 32,
        stain_normalize: bool = True
    ) -> Dict:
        """
        Predict tumor/normal regions in a whole slide image.
        
        Args:
            wsi_path: Path to WSI file (.svs, .tiff, etc.)
            mode: 'text' or 'prototype'
            prompts: Custom prompts
            cancer_type: Cancer type for default prompts
            patch_size: Size of patches to extract
            stride: Stride between patches
            tissue_threshold: Minimum tissue fraction to include patch
            batch_size: Batch size for processing
            stain_normalize: Whether to apply Macenko normalization
        
        Returns:
            Dictionary with slide-level results and patch-level predictions
        """
        try:
            import openslide
            import cv2
        except ImportError:
            raise ImportError("Please install openslide-python and opencv-python")
        
        from ..preprocessing import MacenkoNormalizer
        
        wsi_path = Path(wsi_path)
        if not wsi_path.exists():
            raise FileNotFoundError(f"WSI not found: {wsi_path}")
        
        # Open slide
        slide = openslide.OpenSlide(str(wsi_path))
        width, height = slide.dimensions
        
        # Initialize normalizer
        normalizer = None
        if stain_normalize:
            normalizer = MacenkoNormalizer()
            # Set reference from first valid patch
            for x in range(0, width - patch_size, stride * 4):
                for y in range(0, height - patch_size, stride * 4):
                    patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
                    patch_np = np.array(patch)
                    gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
                    _, tissue = cv2.threshold(gray, 200, 1, cv2.THRESH_BINARY_INV)
                    if tissue.sum() / tissue.size > tissue_threshold:
                        normalizer.fit(patch_np)
                        break
                if normalizer.target_means is not None:
                    break
        
        # Extract patches and predict
        patches = []
        positions = []
        
        for x in range(0, width - patch_size, stride):
            for y in range(0, height - patch_size, stride):
                patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
                patch_np = np.array(patch)
                
                # Tissue check
                gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
                _, tissue = cv2.threshold(gray, 200, 1, cv2.THRESH_BINARY_INV)
                tissue_pct = tissue.sum() / tissue.size
                
                if tissue_pct > tissue_threshold:
                    # Stain normalize
                    if normalizer is not None:
                        try:
                            patch_np = normalizer.transform(patch_np)
                        except:
                            pass
                    
                    patches.append(Image.fromarray(patch_np))
                    positions.append((x, y))
        
        slide.close()
        
        # Predict patches
        results = self.predict_batch(
            patches, mode=mode, prompts=prompts, 
            cancer_type=cancer_type, batch_size=batch_size
        )
        
        # Aggregate results
        tumor_count = sum(1 for r in results if r["prediction"] == "tumor")
        normal_count = len(results) - tumor_count
        tumor_fraction = tumor_count / len(results) if results else 0
        
        avg_tumor_prob = np.mean([r["tumor_probability"] for r in results]) if results else 0.5
        
        return {
            "wsi_path": str(wsi_path),
            "total_patches": len(results),
            "tumor_patches": tumor_count,
            "normal_patches": normal_count,
            "tumor_fraction": tumor_fraction,
            "avg_tumor_probability": avg_tumor_prob,
            "slide_prediction": "tumor" if tumor_fraction > 0.5 else "normal",
            "patch_results": results,
            "positions": positions,
            "mode": mode
        }


def create_inference_pipeline(
    vision_checkpoint: Optional[str] = None,
    text_encoder: str = "Qwen/Qwen2-1.5B",
    device: str = "cuda"
) -> CPathOmniInference:
    """
    Convenience function to create inference pipeline.
    
    Args:
        vision_checkpoint: Path to vision encoder checkpoint
        text_encoder: HuggingFace model name for text encoder
        device: Device to run on
    
    Returns:
        CPathOmniInference instance
    """
    return CPathOmniInference(
        vision_encoder_path=vision_checkpoint,
        text_encoder=text_encoder,
        device=device
    )
