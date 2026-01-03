"""
Training Scripts for CPath-Omni

Includes:
- Linear probe training
- Few-shot fine-tuning
- Adapter fine-tuning
- Semantic Anchoring projection head training
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
import json


class LinearProbe:
    """
    Linear probe classifier on frozen embeddings.
    
    This is the simplest baseline: train a logistic regression
    classifier on top of frozen visual embeddings.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced"
    ):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver='lbfgs',
            random_state=42
        )
    
    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ):
        """Train the linear probe."""
        self.model.fit(embeddings, labels)
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(embeddings)
    
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(embeddings)[:, 1]
    
    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate on test set."""
        predictions = self.predict(embeddings)
        probabilities = self.predict_proba(embeddings)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "auc_roc": roc_auc_score(labels, probabilities)
        }


class FewShotFineTuner:
    """
    Few-shot fine-tuning with a classification head.
    
    Fine-tunes either just a classification head (frozen backbone)
    or the full model with a small learning rate.
    """
    
    def __init__(
        self,
        embedding_dim: int = 3328,
        num_classes: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        freeze_backbone: bool = True,
        device: str = "cuda"
    ):
        self.device = device
        self.freeze_backbone = freeze_backbone
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        ).to(device)
        
        self.backbone = None
    
    def set_backbone(self, backbone: nn.Module):
        """Set the vision backbone for end-to-end fine-tuning."""
        self.backbone = backbone.to(self.device)
        
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def train_on_embeddings(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        val_embeddings: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01
    ) -> Dict[str, List[float]]:
        """Train classification head on pre-computed embeddings."""
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(train_embeddings),
            torch.LongTensor(train_labels)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.head.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        # Loss with class balancing
        class_counts = np.bincount(train_labels)
        class_weights = torch.FloatTensor(
            len(train_labels) / (len(class_counts) * class_counts)
        ).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        history = {"train_loss": [], "val_auc": []}
        best_auc = 0
        
        for epoch in range(epochs):
            self.head.train()
            epoch_loss = 0
            
            for batch_emb, batch_labels in train_loader:
                batch_emb = batch_emb.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                logits = self.head(batch_emb)
                loss = criterion(logits, batch_labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            scheduler.step()
            history["train_loss"].append(epoch_loss / len(train_loader))
            
            # Validation
            if val_embeddings is not None:
                metrics = self.evaluate(val_embeddings, val_labels)
                history["val_auc"].append(metrics["auc_roc"])
                
                if metrics["auc_roc"] > best_auc:
                    best_auc = metrics["auc_roc"]
                    self.best_state = {
                        k: v.cpu().clone() 
                        for k, v in self.head.state_dict().items()
                    }
        
        # Load best model
        if hasattr(self, 'best_state'):
            self.head.load_state_dict(self.best_state)
        
        return history
    
    @torch.no_grad()
    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict on embeddings."""
        self.head.eval()
        
        emb_tensor = torch.FloatTensor(embeddings).to(self.device)
        logits = self.head(emb_tensor)
        probs = F.softmax(logits, dim=1)
        
        predictions = logits.argmax(dim=1).cpu().numpy()
        tumor_probs = probs[:, 1].cpu().numpy()
        
        return predictions, tumor_probs
    
    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate on test set."""
        predictions, tumor_probs = self.predict(embeddings)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "auc_roc": roc_auc_score(labels, tumor_probs)
        }


class AdapterFineTuner:
    """
    Adapter-based fine-tuning for vision transformers.
    
    Adds small adapter modules to each transformer block while
    keeping the original weights frozen. This enables efficient
    fine-tuning with much fewer parameters.
    """
    
    def __init__(
        self,
        model: nn.Module,
        adapter_dim: int = 64,
        num_classes: int = 2,
        device: str = "cuda"
    ):
        self.device = device
        self.model = model.to(device)
        self.adapter_dim = adapter_dim
        
        # Freeze base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Add adapters to transformer blocks
        self.adapters = nn.ModuleList()
        self._add_adapters()
        
        # Classification head
        self.classifier = nn.Linear(
            self._get_embed_dim(), num_classes
        ).to(device)
    
    def _get_embed_dim(self) -> int:
        """Get the embedding dimension from the model."""
        # Try different attribute names
        if hasattr(self.model, 'embed_dim'):
            return self.model.embed_dim
        elif hasattr(self.model, 'visual'):
            return self.model.visual.output_dim
        else:
            return 3328  # Default for CPath-CLIP
    
    def _add_adapters(self):
        """Add adapter modules to transformer blocks."""
        embed_dim = self._get_embed_dim()
        
        # Find transformer blocks
        if hasattr(self.model, 'visual'):
            blocks = self.model.visual.transformer.resblocks
        elif hasattr(self.model, 'blocks'):
            blocks = self.model.blocks
        else:
            raise ValueError("Cannot find transformer blocks in model")
        
        for i, block in enumerate(blocks):
            adapter = nn.Sequential(
                nn.Linear(embed_dim, self.adapter_dim),
                nn.GELU(),
                nn.Linear(self.adapter_dim, embed_dim)
            ).to(self.device)
            
            # Initialize to near-identity
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
            
            self.adapters.append(adapter)
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters."""
        params = list(self.classifier.parameters())
        for adapter in self.adapters:
            params.extend(adapter.parameters())
        return params
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adapters."""
        # This requires modifying the forward pass to include adapters
        # Simplified version - actual implementation depends on model architecture
        features = self.model.encode_image(x)
        logits = self.classifier(features)
        return logits


class PrototypeClassifier:
    """
    Zero-shot prototype-based classifier.
    
    Creates class prototypes from labeled examples and classifies
    new samples based on cosine similarity to prototypes.
    """
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.tumor_prototype = None
        self.normal_prototype = None
    
    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_prototypes: Optional[int] = None
    ):
        """
        Compute class prototypes from labeled embeddings.
        
        Args:
            embeddings: [N, D] feature vectors
            labels: [N] binary labels (0=normal, 1=tumor)
            n_prototypes: If set, use only this many samples per class
        """
        tumor_mask = labels == 1
        normal_mask = labels == 0
        
        tumor_emb = embeddings[tumor_mask]
        normal_emb = embeddings[normal_mask]
        
        if n_prototypes is not None:
            # Random subset
            np.random.seed(42)
            if len(tumor_emb) > n_prototypes:
                idx = np.random.choice(len(tumor_emb), n_prototypes, replace=False)
                tumor_emb = tumor_emb[idx]
            if len(normal_emb) > n_prototypes:
                idx = np.random.choice(len(normal_emb), n_prototypes, replace=False)
                normal_emb = normal_emb[idx]
        
        # Compute mean prototypes
        self.tumor_prototype = tumor_emb.mean(axis=0)
        self.normal_prototype = normal_emb.mean(axis=0)
        
        if self.normalize:
            self.tumor_prototype /= np.linalg.norm(self.tumor_prototype)
            self.normal_prototype /= np.linalg.norm(self.normal_prototype)
        
        return self
    
    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify embeddings based on similarity to prototypes.
        
        Returns:
            predictions: Binary predictions
            tumor_probs: Probability of tumor class
        """
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Cosine similarities
        tumor_sim = embeddings @ self.tumor_prototype
        normal_sim = embeddings @ self.normal_prototype
        
        # Softmax for probabilities
        logits = np.stack([normal_sim, tumor_sim], axis=1)
        exp_logits = np.exp(logits * 100)  # Temperature scaling
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        tumor_probs = probs[:, 1]
        
        predictions = (tumor_sim > normal_sim).astype(int)
        
        return predictions, tumor_probs
    
    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate on test set."""
        predictions, tumor_probs = self.predict(embeddings)
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "auc_roc": roc_auc_score(labels, tumor_probs),
            "prototype_similarity": float(
                self.tumor_prototype @ self.normal_prototype
            )
        }


def run_few_shot_experiment(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    fractions: List[float] = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20],
    n_runs: int = 5,
    device: str = "cuda"
) -> Dict[str, Dict[str, float]]:
    """
    Run few-shot learning experiment with varying amounts of labeled data.
    
    Args:
        train_embeddings: Training embeddings
        train_labels: Training labels
        test_embeddings: Test embeddings
        test_labels: Test labels
        fractions: List of training data fractions to try
        n_runs: Number of random runs per fraction
        device: Computation device
        
    Returns:
        Results dictionary with AUC/accuracy for each fraction
    """
    results = {}
    
    for frac in fractions:
        n_samples = int(len(train_embeddings) * frac)
        print(f"\nFraction: {frac*100:.0f}% ({n_samples} samples)")
        
        run_aucs = []
        run_accs = []
        
        for run in range(n_runs):
            # Random subset
            np.random.seed(run)
            idx = np.random.choice(
                len(train_embeddings), n_samples, replace=False
            )
            
            sub_embeddings = train_embeddings[idx]
            sub_labels = train_labels[idx]
            
            # Train
            trainer = FewShotFineTuner(
                embedding_dim=train_embeddings.shape[1],
                device=device
            )
            trainer.train_on_embeddings(
                sub_embeddings, sub_labels,
                epochs=50, batch_size=min(32, n_samples)
            )
            
            # Evaluate
            metrics = trainer.evaluate(test_embeddings, test_labels)
            run_aucs.append(metrics["auc_roc"])
            run_accs.append(metrics["accuracy"])
        
        results[f"{frac*100:.0f}%"] = {
            "auc_mean": np.mean(run_aucs),
            "auc_std": np.std(run_aucs),
            "acc_mean": np.mean(run_accs),
            "acc_std": np.std(run_accs),
            "n_samples": n_samples
        }
        
        print(f"  AUC: {np.mean(run_aucs)*100:.2f}% ± {np.std(run_aucs)*100:.2f}%")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Training utilities loaded successfully")
    
    # Demo with random data
    np.random.seed(42)
    n_samples = 1000
    embed_dim = 3328
    
    embeddings = np.random.randn(n_samples, embed_dim).astype(np.float32)
    labels = np.random.randint(0, 2, n_samples)
    
    # Linear probe
    print("\nLinear Probe:")
    probe = LinearProbe()
    probe.fit(embeddings[:800], labels[:800])
    metrics = probe.evaluate(embeddings[800:], labels[800:])
    print(f"  AUC: {metrics['auc_roc']*100:.2f}%")
    
    # Prototype classifier
    print("\nPrototype Classifier:")
    proto = PrototypeClassifier()
    proto.fit(embeddings[:800], labels[:800], n_prototypes=50)
    metrics = proto.evaluate(embeddings[800:], labels[800:])
    print(f"  AUC: {metrics['auc_roc']*100:.2f}%")
    print(f"  Prototype similarity: {metrics['prototype_similarity']:.4f}")
