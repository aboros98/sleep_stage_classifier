from typing import Tuple

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE) for sequence modeling."""
    
    def __init__(
        self,
        dim: int,
        max_seq_length: int = 2048,
        theta: float = 1e4
    ) -> None:
        """
        Initialize RoPE layer.
        
        Args:
            dim: Dimension of each attention head.
            max_seq_length: Maximum sequence length to support.
            theta: Base frequency for rotation computation.
        """
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.max_seq_length = max_seq_length
        self.register_buffer("inv_freq", inv_freq)

    def __rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dimensions for position encoding.
        
        Args:
            x: Input tensor.
            
        Returns:
            Rotated tensor.
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]

        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding to query and key tensors.
        
        Args:
            q: Query tensor of shape (batch_size, n_heads, seq_len, head_dim).
            k: Key tensor of shape (batch_size, n_heads, seq_len, head_dim).
            
        Returns:
            Tuple containing position-encoded query and key tensors.
        """
        pos = torch.arange(0, self.max_seq_length, dtype=torch.float32, device=q.device)

        freqs = pos[..., None] @ self.inv_freq[None, ...].to(torch.float32)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos()[None, None, :, :].to(q.dtype)
        sin = emb.sin()[None, None, :, :].to(q.dtype)

        q = (q * cos) + (self.__rotate_half(q) * sin)
        k = (k * cos) + (self.__rotate_half(k) * sin)

        return q, k
