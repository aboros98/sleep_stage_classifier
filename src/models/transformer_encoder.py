from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.training.losses import FocalLoss

from .layers.rope import RoPE
from .layers.transformer_block import TransformerEncoderBlock


class TransformerEncoderModel(nn.Module):
    """Transformer encoder model for sleep stage classification."""
    
    def __init__(
        self,
        n_features: int,
        num_classes: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        intermediate_dim: int = 128,
        num_layers: int = 2,
        max_seq_length: int = 2048,
        dropout_p: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        loss_type: str = "cross_entropy",
        label_smoothing: float = 0.0,
        gamma: float = 2.0,
    ) -> None:
        """
        Initialize the TransformerEncoderModel.
        
        Args:
            n_features: Number of input features.
            num_classes: Number of output classes.
            embed_dim: Embedding dimension for transformer layers.
            n_heads: Number of attention heads.
            intermediate_dim: Dimension of feed-forward intermediate layer.
            num_layers: Number of transformer encoder layers.
            max_seq_length: Maximum sequence length for positional encoding.
            dropout_p: Dropout probability.
            class_weights: Optional weights for each class in loss calculation.
            loss_type: Type of loss function ('cross_entropy' or 'focal').
            label_smoothing: Label smoothing factor for cross-entropy loss.
            gamma: Focusing parameter for focal loss.
        """
        super().__init__()

        self.projection_layer = nn.Linear(n_features, embed_dim)
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                intermediate_dim=intermediate_dim,
                dropout_p=dropout_p
            ) for _ in range(num_layers)
        ])

        self.rope = RoPE(dim=embed_dim // n_heads, max_seq_length=max_seq_length)
        self.output_norm = nn.RMSNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, num_classes)
        self.attention_pool = nn.Linear(embed_dim, 1)
        self._class_weights = class_weights

        if loss_type == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss(
                weight=self._class_weights,
                label_smoothing=label_smoothing,
                reduction="mean"
            )
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(
                gamma=gamma,
                alpha=self._class_weights,
                reduction="mean"
            )
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}. Choose 'cross_entropy' or 'focal'.")

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features).
            targets: Target labels of shape (batch_size,).
            
        Returns:
            Tuple containing:
                - logits: Class predictions of shape (batch_size, num_classes).
                - loss: Scalar loss value.
        """
        x = self.projection_layer(x)

        for layer in self.transformer_layers:
            x = layer(x, self.rope)

        x = self.output_norm(x)
        attn_weights = torch.softmax(self.attention_pool(x), dim=1)
        x = (x * attn_weights).sum(dim=1)

        logits = self.fc1(x)

        if targets is None:
            return logits

        loss = self.loss_fn(logits, targets)

        return logits, loss
