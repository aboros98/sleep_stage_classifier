import torch
import torch.nn as nn

from .attention import Attention
from .mlp import MLP


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with self-attention and feed-forward layers."""
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        intermediate_dim: int,
        dropout_p: float = 0.1
    ) -> None:
        """
        Initialize the TransformerEncoderBlock.
        
        Args:
            embed_dim: Dimension of input embeddings.
            n_heads: Number of attention heads.
            intermediate_dim: Dimension of the feed-forward intermediate layer.
            dropout_p: Dropout probability.
        """
        super().__init__()
        
        self.ln1 = nn.RMSNorm(embed_dim)
        self.attention = Attention(embed_dim, n_heads, dropout_p)
        self.ln2 = nn.RMSNorm(embed_dim)
        self.mlp = MLP(embed_dim, intermediate_dim, dropout_p)

    def forward(self, x: torch.Tensor, pe_layer: nn.Module) -> torch.Tensor:
        """
        Apply transformer block operations.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embed_dim).
            pe_layer: Positional encoding module.
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, embed_dim).
        """
        x = x + self.attention(self.ln1(x), pe_layer)
        x = x + self.mlp(self.ln2(x))
        
        return x
