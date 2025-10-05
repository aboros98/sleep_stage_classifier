from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in classification tasks."""
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
        reduction: str = 'mean'
    ) -> None:
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Per-class weights of shape (num_classes,). If None, all classes are weighted equally.
            gamma: Focusing parameter for hard examples. Higher values focus more on hard examples.
                  Typical values: 0 (equivalent to CE), 1, 2 (default), 5.
            ignore_index: Index to ignore in loss computation (e.g., for padding).
            reduction: Reduction method - 'none', 'mean', or 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Raw logits from model of shape (batch_size, num_classes).
            targets: Ground truth labels of shape (batch_size,).
            
        Returns:
            Focal loss value (scalar if reduction is 'mean' or 'sum').
        """
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            ignore_index=self.ignore_index
        )
        
        p = torch.exp(-ce_loss)
        focal_weight = (1 - p) ** self.gamma
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            alpha_t = self.alpha.gather(0, targets.clamp(min=0))
            alpha_t = torch.where(
                targets != self.ignore_index,
                alpha_t,
                torch.zeros_like(alpha_t)
            )
            loss = alpha_t * focal_weight * ce_loss
        else:
            loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            if self.ignore_index is not None:
                valid_mask = targets != self.ignore_index
                return loss.sum() / valid_mask.sum().clamp(min=1)
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
