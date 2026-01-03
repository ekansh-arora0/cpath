"""
CPath-CLIP Vision Encoder Wrapper

This module provides a clean interface to the CPath-CLIP vision encoder,
a ViT-L/14-336 model fine-tuned on pathology images.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from typing import Optional, Union, List
from pathlib import Path

try:
    import open_clip
except ImportError:
    raise ImportError("Please install open-clip-torch: pip install open-clip-torch")


class CPathCLIPVisionEncoder(nn.Module):
    """
    CPath-CLIP Vision Encoder
    
    A ViT-L/14-336 model for pathology image embedding extraction.
    Uses OpenAI CLIP normalization constants.
    
    Args:
        checkpoint_path: Path to model weights (.pt file)
        device: torch device (cuda/cpu)
        freeze: Whether to freeze encoder weights
    
    Example:
        >>> encoder = CPathCLIPVisionEncoder("checkpoints/cpath_clip.pt")
        >>> image = torch.randn(1, 3, 336, 336)
        >>> embedding = encoder(image)  # Shape: (1, 3328)
    """
    
    # OpenAI CLIP normalization constants
    OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)
    
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Union[str, torch.device] = "cuda",
        freeze: bool = True
    ):
        super().__init__()
        
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Create ViT-L/14-336 backbone
        self.backbone = open_clip.create_model('ViT-L-14-336', pretrained=False)
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            self.backbone.load_state_dict(ckpt, strict=False)
            print(f"✓ Loaded CPath-CLIP checkpoint from {checkpoint_path}")
        
        self.backbone = self.backbone.to(self.device)
        
        # Freeze weights if requested
        if freeze:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Determine embedding dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 336, 336).to(self.device)
            self.embed_dim = self.backbone.encode_image(dummy).shape[1]
        
        print(f"✓ CPath-CLIP initialized, embedding dim: {self.embed_dim}")
        
        # Standard preprocessing transform
        self.preprocess = transforms.Compose([
            transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.OPENAI_MEAN, std=self.OPENAI_STD)
        ])
    
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Extract image embeddings.
        
        Args:
            x: Input tensor of shape (B, 3, 336, 336)
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            Embeddings of shape (B, embed_dim)
        """
        x = x.to(self.device)
        embeddings = self.backbone.encode_image(x)
        
        if normalize:
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        return embeddings
    
    def encode_images(
        self,
        images: List,
        batch_size: int = 32,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode a list of PIL images or numpy arrays.
        
        Args:
            images: List of PIL Images or numpy arrays
            batch_size: Batch size for processing
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            Embeddings tensor of shape (N, embed_dim)
        """
        from PIL import Image
        import numpy as np
        
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Convert to tensors
            tensors = []
            for img in batch:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                tensor = self.preprocess(img)
                tensors.append(tensor)
            
            batch_tensor = torch.stack(tensors).to(self.device)
            
            with torch.no_grad():
                embeddings = self.forward(batch_tensor, normalize=normalize)
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights from the last transformer layer.
        Useful for Grad-CAM visualization.
        
        Args:
            x: Input tensor of shape (B, 3, 336, 336)
        
        Returns:
            Attention weights from the last layer
        """
        x = x.to(self.device)
        
        # Get visual transformer
        visual = self.backbone.visual
        
        # Patch embedding
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        # Add class token and positional embedding
        x = torch.cat([
            visual.class_embedding.expand(x.shape[0], -1, -1),
            x
        ], dim=1)
        x = x + visual.positional_embedding
        
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        
        # Pass through transformer blocks, capture last attention
        attn_weights = None
        for i, block in enumerate(visual.transformer.resblocks):
            if i == len(visual.transformer.resblocks) - 1:
                # Last block - capture attention
                attn_weights = block.attn(block.ln_1(x), need_weights=True)[1]
            x = block(x)
        
        return attn_weights


def load_vision_encoder(
    checkpoint_path: Optional[str] = None,
    device: str = "cuda"
) -> CPathCLIPVisionEncoder:
    """
    Convenience function to load vision encoder.
    
    Args:
        checkpoint_path: Path to checkpoint (optional)
        device: Device to load model on
    
    Returns:
        CPathCLIPVisionEncoder instance
    """
    return CPathCLIPVisionEncoder(
        checkpoint_path=checkpoint_path,
        device=device,
        freeze=True
    )
