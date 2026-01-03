"""
Text Encoder Module: Qwen2-1.5B with Projection Head

This module provides text encoding for pathology prompts using Qwen2-1.5B,
with a learned projection head to align text embeddings with the CPath-CLIP
vision embedding space.
"""

import torch
import torch.nn as nn
from typing import List, Union, Optional
from pathlib import Path


class TextProjectionHead(nn.Module):
    """
    MLP projection head to align text embeddings with vision space.
    
    Projects Qwen embeddings (1536-dim) to CPath-CLIP vision space (3328-dim).
    
    Architecture:
        Linear(1536, 2048) -> LayerNorm -> GELU ->
        Linear(2048, 2048) -> LayerNorm -> GELU ->
        Linear(2048, 3328)
    
    Args:
        input_dim: Dimension of text encoder output (default: 1536 for Qwen2-1.5B)
        hidden_dim: Hidden layer dimension (default: 2048)
        output_dim: Output dimension to match vision encoder (default: 3328)
    """
    
    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 2048,
        output_dim: int = 3328
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project text embeddings to vision space.
        
        Args:
            x: Text embeddings of shape (B, input_dim)
        
        Returns:
            Projected embeddings of shape (B, output_dim)
        """
        return self.projection(x)


class QwenTextEncoder(nn.Module):
    """
    Qwen2-1.5B Text Encoder for Pathology Prompts
    
    Uses Qwen2-1.5B as a text encoder with mean pooling over sequence length,
    followed by a projection head to align with CPath-CLIP vision embeddings.
    
    Args:
        model_name: HuggingFace model name (default: "Qwen/Qwen2-1.5B")
        projection_path: Optional path to pretrained projection weights
        device: torch device
        dtype: Model dtype (default: float16 for efficiency)
    
    Example:
        >>> encoder = QwenTextEncoder(device="cuda")
        >>> prompts = ["tumor tissue with nuclear atypia", "normal tissue"]
        >>> embeddings = encoder.encode_text(prompts)  # Shape: (2, 3328)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B",
        projection_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float16,
        vision_dim: int = 3328
    ):
        super().__init__()
        
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        
        print(f"Loading {model_name}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        
        # Determine text embedding dimension
        self.text_dim = self.model.config.hidden_size  # 1536 for Qwen2-1.5B
        
        # Create projection head
        self.projection = TextProjectionHead(
            input_dim=self.text_dim,
            output_dim=vision_dim
        ).to(self.device)
        
        # Load pretrained projection if provided
        if projection_path is not None:
            projection_path = Path(projection_path)
            if projection_path.exists():
                state_dict = torch.load(projection_path, map_location=self.device)
                self.projection.load_state_dict(state_dict)
                print(f"✓ Loaded projection weights from {projection_path}")
        
        print(f"✓ Qwen text encoder initialized")
        print(f"  Text dim: {self.text_dim}, Vision dim: {vision_dim}")
    
    def forward(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode text prompts.
        
        Args:
            texts: Single string or list of strings
            normalize: Whether to L2-normalize output embeddings
        
        Returns:
            Text embeddings of shape (B, vision_dim)
        """
        return self.encode_text(texts, normalize=normalize)
    
    @torch.no_grad()
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        max_length: int = 512
    ) -> torch.Tensor:
        """
        Encode text prompts to vision-aligned embeddings.
        
        Args:
            texts: Single string or list of strings
            normalize: Whether to L2-normalize output embeddings
            max_length: Maximum sequence length
        
        Returns:
            Text embeddings of shape (B, vision_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Encode
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # Mean pool over sequence length
        text_emb = outputs.last_hidden_state.mean(dim=1).float()
        
        # Project to vision space
        text_emb = self.projection(text_emb)
        
        # Normalize if requested
        if normalize:
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        
        return text_emb
    
    def save_projection(self, path: Union[str, Path]):
        """Save projection head weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.projection.state_dict(), path)
        print(f"✓ Saved projection weights to {path}")


# Predefined pathology prompts for different cancer types
PATHOLOGY_PROMPTS = {
    "breast_cancer": {
        "tumor": "Histopathological image showing invasive breast carcinoma with irregular glandular structures, nuclear pleomorphism, and increased mitotic activity characteristic of malignant epithelial proliferation.",
        "normal": "Histopathological image showing normal breast tissue with regular ductal and lobular architecture, uniform nuclei, and organized stromal components without signs of malignancy."
    },
    "mast_cell_tumor": {
        "tumor": "Histopathological image showing mast cell tumor with sheets of round cells containing granular cytoplasm, eccentric nuclei, and characteristic metachromatic granules.",
        "normal": "Histopathological image showing normal connective tissue with scattered mast cells, organized collagen fibers, and no evidence of neoplastic proliferation."
    },
    "generic": {
        "tumor": "Histopathological image showing tumor tissue with cellular atypia, architectural disorganization, and features of malignancy.",
        "normal": "Histopathological image showing normal tissue with preserved architecture and no signs of malignancy."
    }
}


def get_prompts(cancer_type: str = "generic") -> dict:
    """
    Get predefined prompts for a cancer type.
    
    Args:
        cancer_type: One of 'breast_cancer', 'mast_cell_tumor', 'generic'
    
    Returns:
        Dictionary with 'tumor' and 'normal' prompts
    """
    if cancer_type not in PATHOLOGY_PROMPTS:
        print(f"Warning: Unknown cancer type '{cancer_type}', using generic prompts")
        cancer_type = "generic"
    return PATHOLOGY_PROMPTS[cancer_type]
