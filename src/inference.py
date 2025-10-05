import os
from typing import List, Optional, Union

import joblib
import numpy as np
import torch

from src.configs import get_config
from src.data import preprocess_for_inference
from src.models import LSTMClassifier
from src.models.transformer_encoder import TransformerEncoderModel


class SklearnModelWrapper:
    """Wrapper for sklearn models to provide consistent interface with PyTorch models."""
    
    def __init__(self, sklearn_model):
        """
        Initialize wrapper.
        
        Args:
            sklearn_model: Trained sklearn model (e.g., LogisticRegression).
        """
        self.model = sklearn_model
    
    def eval(self):
        """Compatibility method (sklearn models don't have eval mode)."""
        pass
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass for sklearn model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_features).
            targets: Unused, for compatibility with PyTorch models.
            
        Returns:
            Tuple of (logits, None) where logits are class probabilities.
        """
        # Flatten the sequence dimension for sklearn
        # Shape: (batch_size, sequence_length, n_features) -> (batch_size, sequence_length * n_features)
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1).cpu().numpy()
        
        # Get predictions and probabilities
        proba = self.model.predict_proba(x_flat)
        
        # Convert to torch tensor (logits are log probabilities for compatibility)
        logits = torch.tensor(proba, dtype=torch.float32)
        
        return logits
    
    def to(self, device):
        """Compatibility method (sklearn models don't need device placement)."""
        return self


class SleepStagePredictor:
    """
    Simple inference class for sleep stage classification.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the sleep stage predictor.
        
        Args:
            checkpoint_path: Path to the model checkpoint (.ckpt file).
            device: Device to run inference on ('cuda' or 'cpu').
        """
        self.device = device
        self.config = get_config()
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
    
    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            
        Returns:
            Loaded model.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # For logistic regression, load sklearn model
        if self.config.model_type == "logistic" or checkpoint_path.endswith('.pkl'):
            print("Loading Logistic Regression model...")
            model = joblib.load(checkpoint_path)
            # Wrap sklearn model to have a consistent interface
            return SklearnModelWrapper(model)
        
        # For neural network models (LSTM, Transformer)
        # Initialize model based on config
        if self.config.model_type == "lstm":
            model = LSTMClassifier(
                n_features=self.config.n_features,
                hidden_dim=self.config.hidden_dim,
                num_classes=self.config.num_classes,
                num_lstm_layers=self.config.num_lstm_layers,
                conv_out_channels=self.config.conv_out_channels,
                dropout_p=self.config.dropout_p,
                loss_type=self.config.loss_type,
                gamma=self.config.focal_gamma,
                label_smoothing=self.config.label_smoothing,
                conv1_kernel_size=self.config.conv1_kernel_size,
                conv1_stride=self.config.conv1_stride,
                conv1_padding=self.config.conv1_padding,
                conv2_kernel_size=self.config.conv2_kernel_size,
                conv2_stride=self.config.conv2_stride,
                conv2_padding=self.config.conv2_padding,
                conv3_kernel_size=self.config.conv3_kernel_size,
                conv3_stride=self.config.conv3_stride,
                conv3_padding=self.config.conv3_padding,
                conv_channel_multiplier=self.config.conv_channel_multiplier,
                lstm_dropout=self.config.lstm_dropout,
                lstm_bidirectional=self.config.lstm_bidirectional,
            )
        elif self.config.model_type == "transformer":
            model = TransformerEncoderModel(
                n_features=self.config.n_features,
                num_classes=self.config.num_classes,
                embed_dim=self.config.transformer_embed_dim,
                n_heads=self.config.transformer_n_heads,
                intermediate_dim=self.config.transformer_intermediate_dim,
                num_layers=self.config.transformer_num_layers,
                max_seq_length=self.config.transformer_max_seq_length,
                dropout_p=self.config.transformer_dropout_p,
                loss_type=self.config.loss_type,
                label_smoothing=self.config.label_smoothing,
                gamma=self.config.focal_gamma,
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Remove 'model.' prefix if present (from PyTorch Lightning)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "") if key.startswith("model.") else key
            new_state_dict[new_key] = value
        
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        
        return model
    
    def predict(
        self,
        hr_data: Union[List[float], np.ndarray],
        accel_data: Union[List[List[float]], np.ndarray],
        duration_seconds: int,
    ) -> Union[np.ndarray, tuple]:
        """
        Run inference on raw sensor data.
        
        Args:
            hr_data: List or array of heart rate values (BPM). Example: [65, 68, 70, ...]
            accel_data: List or array of [x, y, z] accelerometer values. 
                       Example: [[0.1, 0.2, 9.8], [0.2, 0.1, 9.7], ...]
            duration_seconds: Total duration in seconds
            return_probabilities: If True, also return class probabilities.
            batch_size: Batch size for inference (to manage memory).
            
        Returns:
            predictions: Array of predicted sleep stages (0=Wake, 1=NREM, 2=REM)
            probabilities (optional): Array of class probabilities, shape (n_windows, 3)
        """
        # Convert to numpy arrays
        hr_data = np.array(hr_data) if not isinstance(hr_data, np.ndarray) else hr_data
        accel_data = np.array(accel_data) if not isinstance(accel_data, np.ndarray) else accel_data
        
        # Preprocess using preprocess_for_inference
        features = preprocess_for_inference(
            hr_data=hr_data,
            motion_data=accel_data,
            target_duration_seconds=duration_seconds,
            hr_window_sizes=self.config.hr_window_sizes,
            motion_gravity_constant=self.config.motion_gravity_constant,
            normalization_epsilon=self.config.normalization_epsilon,
            min_samples_for_std=self.config.min_samples_for_std,
        )

        features = features[None, -512 :, :] # Remove first 512 seconds to match lookback window
        features = torch.tensor(features, dtype=torch.float32)
        
        # Check if sklearn model
        is_sklearn = isinstance(self.model, SklearnModelWrapper)
        
        with torch.no_grad():
            if not is_sklearn:
                features = features.to(self.device)
                
                # Get model output (only logits, no loss computation)
                logits = self.model.forward(features)
                logits = logits.softmax(dim=-1)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1).numpy()
        
        return predictions