import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Multi-head self-attention layer with positional encoding support."""
    
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        """
        Initialize the Attention layer.
        
        Args:
            embed_dim: Dimension of input embeddings.
            n_heads: Number of attention heads.
            dropout: Dropout probability for attention weights.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.dropout = nn.Dropout(dropout)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, pe_layer: nn.Module) -> torch.Tensor:
        """
        Apply multi-head self-attention with positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embed_dim).
            pe_layer: Positional encoding module (e.g., RoPE).
            
        Returns:
            Attention output tensor of shape (batch_size, sequence_length, embed_dim).
        """
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)

        q = q.view(*x.shape[:-1], self.n_heads, -1).transpose(-2, -3)
        k = k.view(*x.shape[:-1], self.n_heads, -1).transpose(-2, -3)
        v = v.view(*x.shape[:-1], self.n_heads, -1).transpose(-2, -3)
        
        # Apply positional encoding
        q, k = pe_layer(q, k)

        output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p, is_causal=False)
        output = output.transpose(-2, -3).contiguous().view(*x.shape[:-1], -1)

        output = self.out_proj(output)

        return output
