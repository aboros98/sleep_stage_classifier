from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.losses import FocalLoss


class LSTMClassifier(nn.Module):
    """LSTM-based classifier with CNN feature extraction for sleep stage classification."""
    
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        num_classes: int = 3,
        num_lstm_layers: int = 1,
        conv_out_channels: int = 32,
        dropout_p: float = 0.3,
        loss_type: Literal["cross_entropy", "focal"] = "cross_entropy",
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        conv1_kernel_size: int = 7,
        conv1_stride: int = 4,
        conv1_padding: int = 3,
        conv2_kernel_size: int = 5,
        conv2_stride: int = 4,
        conv2_padding: int = 2,
        conv3_kernel_size: int = 3,
        conv3_stride: int = 3,
        conv3_padding: int = 1,
        conv_channel_multiplier: int = 2,
        lstm_dropout: float = 0.0,
        lstm_bidirectional: bool = True,
    ) -> None:
        """
        Initialize the LSTMClassifier.
        
        Args:
            n_features: Number of input features per timestep.
            hidden_dim: Hidden dimension size for LSTM layers.
            num_classes: Number of output classes.
            num_lstm_layers: Number of stacked LSTM layers.
            conv_out_channels: Number of output channels for convolutional layers.
            dropout_p: Dropout probability.
            loss_type: Type of loss function ('cross_entropy' or 'focal').
            class_weights: Optional weights for each class in loss calculation.
            gamma: Focusing parameter for focal loss.
            label_smoothing: Label smoothing factor for cross-entropy loss.
            conv1_kernel_size: Kernel size for first conv layer.
            conv1_stride: Stride for first conv layer.
            conv1_padding: Padding for first conv layer.
            conv2_kernel_size: Kernel size for second conv layer.
            conv2_stride: Stride for second conv layer.
            conv2_padding: Padding for second conv layer.
            conv3_kernel_size: Kernel size for third conv layer.
            conv3_stride: Stride for third conv layer.
            conv3_padding: Padding for third conv layer.
            conv_channel_multiplier: Multiplier for conv channel expansion.
            lstm_dropout: Dropout for LSTM layers.
            lstm_bidirectional: Whether to use bidirectional LSTM.
        """
        super().__init__()
        
        # Multi-scale downsampling
        self.conv1 = nn.Conv1d(n_features, conv_out_channels, 
                               kernel_size=conv1_kernel_size, 
                               stride=conv1_stride, 
                               padding=conv1_padding)
        self.bn1 = nn.BatchNorm1d(conv_out_channels)
        
        self.conv2 = nn.Conv1d(conv_out_channels, conv_out_channels * conv_channel_multiplier, 
                               kernel_size=conv2_kernel_size, 
                               stride=conv2_stride, 
                               padding=conv2_padding)
        self.bn2 = nn.BatchNorm1d(conv_out_channels * conv_channel_multiplier)
        
        self.conv3 = nn.Conv1d(conv_out_channels * conv_channel_multiplier, 
                               conv_out_channels * conv_channel_multiplier, 
                               kernel_size=conv3_kernel_size, 
                               stride=conv3_stride, 
                               padding=conv3_padding)
        self.bn3 = nn.BatchNorm1d(conv_out_channels * conv_channel_multiplier)
        
        self.dropout_conv = nn.Dropout(dropout_p)
        
        lstm_hidden_multiplier = 2 if lstm_bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=conv_out_channels * conv_channel_multiplier,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout
        )
        
        self.attention = nn.Linear(hidden_dim * lstm_hidden_multiplier, 1)
        self.dropout_fc = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim * lstm_hidden_multiplier, num_classes)
        
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
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the LSTM classifier.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features).
            targets: Target labels of shape (batch_size,).
        
        Returns:
            Tuple containing:
                - logits: Class predictions of shape (batch_size, num_classes).
                - loss: Scalar loss value.
        """
        # Transpose for Conv1d: (B, seq_len, features) -> (B, features, seq_len)
        x = x.transpose(1, 2)
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout_conv(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout_conv(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout_conv(x)
        
        # Transpose back for LSTM: (B, features, seq_len) -> (B, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM processing
        x, _ = self.lstm(x)
        
        # Attention pooling
        attention_weights = torch.softmax(self.attention(x), dim=1)
        x = (x * attention_weights).sum(dim=1)
        
        # Classification
        x = self.dropout_fc(x)
        logits = self.fc(x)

        if targets is None:
            return logits
        
        loss = self.loss_fn(logits, targets)
        
        return logits, loss
