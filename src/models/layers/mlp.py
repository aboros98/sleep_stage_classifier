import torch
import torch.nn as nn


class MLP(nn.Module):
    """Gated MLP with SwiGLU activation for transformer blocks."""
    
    def __init__(self, embed_dim: int, intermediate_dim: int, dropout_p: float = 0.2) -> None:
        """
        Initialize the MLP layer.
        
        Args:
            embed_dim: Dimension of input/output embeddings.
            intermediate_dim: Dimension of the intermediate hidden layer.
            dropout_p: Dropout probability.
        """
        super(MLP, self).__init__()

        self.up_proj = nn.Linear(embed_dim, intermediate_dim)
        self.gate_proj = nn.Linear(embed_dim, intermediate_dim)
        self.down_proj = nn.Linear(intermediate_dim, embed_dim)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply gated MLP transformation.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embed_dim).
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, embed_dim).
        """
        return self.down_proj(self.dropout(self.act(self.gate_proj(x)) * self.up_proj(x)))
