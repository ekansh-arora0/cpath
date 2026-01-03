"""
Semantic Anchoring Module

Core implementation of the Semantic Anchoring method that uses language to
provide stable classification anchors across domain shifts.

The key insight is that while visual prototypes shift between domains (species/cancer types),
text descriptions of "tumor" vs "normal" remain semantically stable, enabling cross-domain transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class SemanticAnchoringConfig:
    """Configuration for Semantic Anchoring."""
    vision_dim: int = 3328  # CPath-CLIP embedding dimension
    text_dim: int = 768  # CLIP text encoder dimension (or 1536 for Qwen)
    hidden_dim: int = 2048  # Projection head hidden dimension
    temperature: float = 0.07  # Contrastive learning temperature
    use_learned_temperature: bool = False
    normalize_embeddings: bool = True


class TextProjectionHead(nn.Module):
    """
    Projects text embeddings into the visual embedding space.
    
    Architecture: text_dim → hidden → hidden → vision_dim
    With LayerNorm and GELU activations for stable training.
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        vision_dim: int = 3328,
        hidden_dim: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vision_dim)
        )
        
        # Initialize final layer with small weights
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SemanticAnchoring(nn.Module):
    """
    Semantic Anchoring for cross-domain pathology classification.
    
    Instead of comparing visual embeddings to visual prototypes (which shift
    across domains), we compare to text anchors projected into visual space.
    
    Key components:
    1. Text encoder (CLIP or Qwen) - extracts semantic features from prompts
    2. Projection head - aligns text embeddings to visual space
    3. Similarity computation - classifies based on cosine similarity
    
    Usage:
        model = SemanticAnchoring(text_encoder, projection_head)
        
        # Set class anchors
        model.set_anchors({
            "tumor": "Histopathological image showing malignant tumor tissue",
            "normal": "Histopathological image showing normal tissue architecture"
        })
        
        # Classify visual embeddings
        predictions = model.classify(visual_embeddings)
    """
    
    def __init__(
        self,
        text_encoder: nn.Module,
        projection_head: Optional[nn.Module] = None,
        config: Optional[SemanticAnchoringConfig] = None
    ):
        super().__init__()
        
        self.config = config or SemanticAnchoringConfig()
        self.text_encoder = text_encoder
        
        # Create projection head if not provided
        if projection_head is None:
            self.projection = TextProjectionHead(
                text_dim=self.config.text_dim,
                vision_dim=self.config.vision_dim,
                hidden_dim=self.config.hidden_dim
            )
        else:
            self.projection = projection_head
        
        # Learned temperature (optional)
        if self.config.use_learned_temperature:
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(self.config.temperature))
            )
        else:
            self.register_buffer(
                'log_temperature',
                torch.log(torch.tensor(self.config.temperature))
            )
        
        # Class anchors (text embeddings projected to visual space)
        self.register_buffer('tumor_anchor', None)
        self.register_buffer('normal_anchor', None)
        self._anchor_prompts = {}
    
    @property
    def temperature(self) -> float:
        return torch.exp(self.log_temperature).item()
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using the text encoder."""
        # This should be overridden based on the specific text encoder
        # Default assumes text_encoder has an encode method
        with torch.no_grad():
            text_emb = self.text_encoder.encode(text)
        return text_emb
    
    def project_text(self, text_embedding: torch.Tensor) -> torch.Tensor:
        """Project text embedding into visual space."""
        projected = self.projection(text_embedding)
        
        if self.config.normalize_embeddings:
            projected = F.normalize(projected, dim=-1)
        
        return projected
    
    def set_anchors(
        self,
        prompts: Dict[str, str],
        text_embeddings: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Set the classification anchors from text prompts.
        
        Args:
            prompts: Dictionary mapping class names to text prompts
                     {"tumor": "...", "normal": "..."}
            text_embeddings: Pre-computed text embeddings (optional)
        """
        self._anchor_prompts = prompts
        
        if text_embeddings is not None:
            # Use provided embeddings
            tumor_emb = text_embeddings["tumor"]
            normal_emb = text_embeddings["normal"]
        else:
            # Encode prompts
            tumor_emb = self.encode_text(prompts["tumor"])
            normal_emb = self.encode_text(prompts["normal"])
        
        # Project to visual space
        self.tumor_anchor = self.project_text(tumor_emb)
        self.normal_anchor = self.project_text(normal_emb)
    
    def compute_similarities(
        self,
        visual_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cosine similarities between visual embeddings and anchors.
        
        Args:
            visual_embeddings: [batch, vision_dim] visual features
            
        Returns:
            tumor_sim: [batch] similarities to tumor anchor
            normal_sim: [batch] similarities to normal anchor
        """
        if self.tumor_anchor is None or self.normal_anchor is None:
            raise ValueError("Anchors not set. Call set_anchors() first.")
        
        # Normalize visual embeddings
        if self.config.normalize_embeddings:
            visual_embeddings = F.normalize(visual_embeddings, dim=-1)
        
        # Compute cosine similarities
        tumor_sim = F.cosine_similarity(
            visual_embeddings,
            self.tumor_anchor.expand(visual_embeddings.shape[0], -1)
        )
        normal_sim = F.cosine_similarity(
            visual_embeddings,
            self.normal_anchor.expand(visual_embeddings.shape[0], -1)
        )
        
        return tumor_sim, normal_sim
    
    def classify(
        self,
        visual_embeddings: torch.Tensor,
        return_probs: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Classify visual embeddings using semantic anchors.
        
        Args:
            visual_embeddings: [batch, vision_dim] visual features
            return_probs: Whether to return probabilities (softmax)
            
        Returns:
            Dictionary with predictions, probabilities, and similarities
        """
        tumor_sim, normal_sim = self.compute_similarities(visual_embeddings)
        
        # Stack for softmax: [batch, 2] where dim 0 = normal, dim 1 = tumor
        logits = torch.stack([normal_sim, tumor_sim], dim=1) / self.temperature
        
        if return_probs:
            probs = F.softmax(logits * 100, dim=1)  # Scale for numerical stability
            tumor_probs = probs[:, 1]
        else:
            tumor_probs = tumor_sim
        
        # Predictions: 1 = tumor, 0 = normal
        predictions = (tumor_sim > normal_sim).long()
        
        return {
            "predictions": predictions,
            "tumor_probability": tumor_probs,
            "tumor_similarity": tumor_sim,
            "normal_similarity": normal_sim,
            "logits": logits
        }
    
    def forward(
        self,
        visual_embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training or inference.
        
        Args:
            visual_embeddings: [batch, vision_dim] visual features
            labels: [batch] ground truth labels (0=normal, 1=tumor) for training
            
        Returns:
            Dictionary with predictions and optionally loss
        """
        results = self.classify(visual_embeddings)
        
        if labels is not None:
            # Compute InfoNCE-style contrastive loss
            loss = self.contrastive_loss(
                visual_embeddings, labels
            )
            results["loss"] = loss
        
        return results
    
    def contrastive_loss(
        self,
        visual_embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss for training the projection head.
        
        Args:
            visual_embeddings: [batch, vision_dim] visual features
            labels: [batch] ground truth labels (0=normal, 1=tumor)
        """
        tumor_sim, normal_sim = self.compute_similarities(visual_embeddings)
        
        # Stack logits: [batch, 2] where 0=normal, 1=tumor
        logits = torch.stack([normal_sim, tumor_sim], dim=1) / self.temperature
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels.long())
        
        return loss


class SemanticAnchoringTrainer:
    """
    Trainer for the Semantic Anchoring projection head.
    
    Uses contrastive learning to align text anchors with visual embeddings
    from the training set.
    """
    
    def __init__(
        self,
        model: SemanticAnchoring,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.device = device
        
        # Only train the projection head, freeze text encoder
        for param in self.model.text_encoder.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.AdamW(
            self.model.projection.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.warmup_steps = warmup_steps
        self.global_step = 0
    
    def train_epoch(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Shuffle data
        indices = np.random.permutation(len(train_embeddings))
        train_embeddings = train_embeddings[indices]
        train_labels = train_labels[indices]
        
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(train_embeddings), batch_size):
            batch_emb = torch.FloatTensor(
                train_embeddings[i:i+batch_size]
            ).to(self.device)
            batch_labels = torch.LongTensor(
                train_labels[i:i+batch_size]
            ).to(self.device)
            
            # Forward
            results = self.model(batch_emb, batch_labels)
            loss = results["loss"]
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.projection.parameters(), 1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1
        
        return {"loss": total_loss / n_batches}
    
    @torch.no_grad()
    def evaluate(
        self,
        test_embeddings: np.ndarray,
        test_labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate on test set."""
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        self.model.eval()
        
        test_tensor = torch.FloatTensor(test_embeddings).to(self.device)
        results = self.model.classify(test_tensor)
        
        predictions = results["predictions"].cpu().numpy()
        tumor_probs = results["tumor_probability"].cpu().numpy()
        
        auc = roc_auc_score(test_labels, tumor_probs)
        acc = accuracy_score(test_labels, predictions)
        
        return {
            "auc_roc": auc,
            "accuracy": acc
        }
    
    def train(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        val_embeddings: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 10
    ) -> Dict[str, List[float]]:
        """Full training loop with validation."""
        history = {
            "train_loss": [],
            "val_auc": [],
            "val_acc": []
        }
        
        best_auc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(
                train_embeddings, train_labels, batch_size
            )
            history["train_loss"].append(train_metrics["loss"])
            
            # Validate
            if val_embeddings is not None:
                val_metrics = self.evaluate(val_embeddings, val_labels)
                history["val_auc"].append(val_metrics["auc_roc"])
                history["val_acc"].append(val_metrics["accuracy"])
                
                # Early stopping
                if val_metrics["auc_roc"] > best_auc:
                    best_auc = val_metrics["auc_roc"]
                    patience_counter = 0
                    # Save best model
                    self.best_state = {
                        k: v.cpu().clone() 
                        for k, v in self.model.projection.state_dict().items()
                    }
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}: loss={train_metrics['loss']:.4f}, "
                          f"val_auc={val_metrics['auc_roc']*100:.2f}%")
        
        # Load best model
        if hasattr(self, 'best_state'):
            self.model.projection.load_state_dict(self.best_state)
        
        return history


# Default prompts for different cancer types
DEFAULT_PROMPTS = {
    "breast_cancer": {
        "generic": {
            "tumor": "Histopathological image of malignant tumor tissue with abnormal cellular proliferation",
            "normal": "Histopathological image of normal healthy tissue with regular cellular architecture"
        },
        "histological": {
            "tumor": "Invasive carcinoma with nuclear atypia, high mitotic rate, and stromal invasion",
            "normal": "Benign breast parenchyma with organized ductal and lobular structures"
        },
        "snomed_ct": {
            "tumor": "Malignant neoplasm of breast (SNOMED 254837009) showing infiltrating ductal carcinoma",
            "normal": "Normal breast tissue (SNOMED 76752008) with unremarkable glandular architecture"
        }
    },
    "mast_cell_tumor": {
        "generic": {
            "tumor": "Mast cell tumor with neoplastic round cells and metachromatic granules",
            "normal": "Normal dermis with sparse mast cells and organized collagen fibers"
        },
        "histological": {
            "tumor": "Sheets of round cells with abundant cytoplasmic granules and anisokaryosis",
            "normal": "Fibrous connective tissue with scattered inflammatory cells"
        }
    }
}


def get_default_prompts(
    cancer_type: str = "breast_cancer",
    prompt_style: str = "generic"
) -> Dict[str, str]:
    """Get default prompts for a cancer type and style."""
    if cancer_type not in DEFAULT_PROMPTS:
        raise ValueError(f"Unknown cancer type: {cancer_type}")
    
    prompts = DEFAULT_PROMPTS[cancer_type]
    
    if prompt_style not in prompts:
        raise ValueError(f"Unknown prompt style: {prompt_style}")
    
    return prompts[prompt_style]
