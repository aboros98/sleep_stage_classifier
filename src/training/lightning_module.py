from typing import Dict, List, Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.optim as optimizers
import torchmetrics
from torch.utils.data import DataLoader

from src.data import SleepDataset, TrainValSplitter
from src.models import LSTMClassifier
from src.models.transformer_encoder import TransformerEncoderModel

from .scheduler import CosineLinearWarmupLR


class TrainingModel(L.LightningModule):
    """PyTorch Lightning module for sleep stage classification training."""
    
    def __init__(
        self,
        config: dict,
        artifacts_path: str,
        train_subjects_id: Optional[List[int]] = None,
        val_subjects_id: Optional[List[int]] = None
    ) -> None:
        """
        Initialize the TrainingModel.
        
        Args:
            config: Configuration dictionary with training parameters.
            artifacts_path: Path to save training artifacts.
            train_subjects_id: List of subject IDs for training (optional).
            val_subjects_id: List of subject IDs for validation (optional).

        Returns:
            None
        """
        super().__init__()
        self.config = config
        self.training_step_outputs = []
        self.validation_step_outputs = []

        if train_subjects_id is None or val_subjects_id is None:
            splitter = TrainValSplitter(
                main_data_path=self.config.main_data_path,
                val_split=self.config.val_split,
                random_seed=self.config.seed
            )
            train_subjects_id, val_subjects_id = splitter.split()
        
        self.train_dataset = SleepDataset(
            main_data_path=self.config.main_data_path,
            subjects_id=train_subjects_id,
            artifacts_path=artifacts_path,
            lookback_window=self.config.lookback_window,
            n_jobs=self.config.n_jobs,
            hr_window_sizes=self.config.hr_window_sizes,
            motion_gravity_constant=self.config.motion_gravity_constant,
            normalization_epsilon=self.config.normalization_epsilon,
            unknown_label=self.config.unknown_label,
            label_merge_value=self.config.label_merge_n3_n4_to_nrem,
            label_rem_value=self.config.label_rem_remapped,
            psg_buffer=self.config.psg_buffer,
            min_samples_for_std=self.config.min_samples_for_std,
        )

        self.val_dataset = SleepDataset(
            main_data_path=self.config.main_data_path,
            subjects_id=val_subjects_id,
            artifacts_path=artifacts_path,
            lookback_window=self.config.lookback_window,
            n_jobs=self.config.n_jobs,
            hr_window_sizes=self.config.hr_window_sizes,
            motion_gravity_constant=self.config.motion_gravity_constant,
            normalization_epsilon=self.config.normalization_epsilon,
            unknown_label=self.config.unknown_label,
            label_merge_value=self.config.label_merge_n3_n4_to_nrem,
            label_rem_value=self.config.label_rem_remapped,
            psg_buffer=self.config.psg_buffer,
            min_samples_for_std=self.config.min_samples_for_std,
        )

        class_weights = self._compute_class_weights() if self.config.use_class_weights else None

        # Select model based on config
        if self.config.model_type == "lstm":
            self.model = LSTMClassifier(
                n_features=self.train_dataset.n_features,
                hidden_dim=self.config.hidden_dim,
                num_classes=self.config.num_classes,
                num_lstm_layers=self.config.num_lstm_layers,
                conv_out_channels=self.config.conv_out_channels,
                dropout_p=self.config.dropout_p,
                class_weights=class_weights,
                gamma=self.config.focal_gamma,
                loss_type=self.config.loss_type,
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
            self.model = TransformerEncoderModel(
                n_features=self.train_dataset.n_features,
                num_classes=self.config.num_classes,
                embed_dim=self.config.transformer_embed_dim,
                n_heads=self.config.transformer_n_heads,
                intermediate_dim=self.config.transformer_intermediate_dim,
                num_layers=self.config.transformer_num_layers,
                max_seq_length=self.config.transformer_max_seq_length,
                dropout_p=self.config.transformer_dropout_p,
                class_weights=class_weights,
                loss_type=self.config.loss_type,
                label_smoothing=self.config.label_smoothing,
                gamma=self.config.focal_gamma,
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.config.model_type}. Choose 'lstm' or 'transformer'.")

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.train_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.config.num_classes, average='weighted')
        
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.config.num_classes)
        self.val_f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.config.num_classes, average='weighted')
        self.val_f1_per_class = torchmetrics.F1Score(task="multiclass", num_classes=self.config.num_classes, average='none')
  
        if self.config.scheduler:
            self._update_scheduler_config()

    def _update_scheduler_config(self) -> None:
        """
        Update the learning rate scheduler configuration.
        """
        dataset_length = len(self.train_dataset)
        model_steps = dataset_length // self.config.batch_size

        total_training_steps = self.config.max_epochs * model_steps

        warmup_steps = self.config.warmup_percent * total_training_steps

        # Update the warmup steps in config for scheduler
        self.config.warmup_steps = warmup_steps
        self.config.total_training_steps = total_training_steps

    def _compute_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.

        Returns:
            torch.Tensor: A tensor containing the class weights.
        """
        all_labels = self.train_dataset.labels.flatten()
        class_counts = np.bincount(all_labels, minlength=self.config.num_classes)
        
        beta = self.config.class_weight_beta
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / (effective_num + self.config.class_weight_epsilon)
        weights = weights / weights.mean()
        
        return torch.FloatTensor(weights)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.
            target (torch.Tensor, optional): Target tensor for computing loss.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Model output and loss (if target is provided).
        """
        if target is None:
            return self.model(x)

        return self.model(x, target)

    def training_step(self, batch: list, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Training step for a single batch.

        Args:
            batch (list): A list containing the input and target tensors.
            batch_idx (int): The index of the current batch.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the loss and metrics.
        """
        source, target = batch

        logits, loss = self.model(source, target)
        preds = torch.argmax(logits, dim=-1)
        self.train_accuracy(preds, target)
        self.train_f1_score(preds, target)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("train/f1_score", self.train_f1_score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # store for confusion matrix callback (no extra forward pass)
        self.training_step_outputs.append({
            'preds': preds.detach(),
            'targets': target.detach()
        })
        return {"loss": loss}

    def validation_step(self, batch: list, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Validation step for a single batch.

        Args:
            batch (list): A list containing the input and target tensors.
            batch_idx (int): The index of the current batch.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the loss and metrics.
        """
        source, target = batch
        logits, loss = self.model(source, target)

        preds = torch.argmax(logits, dim=-1)

        self.val_accuracy(preds, target)
        self.val_f1_score(preds, target)
        self.val_f1_per_class(preds, target)

        self.log("val/f1_score", self.val_f1_score, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        f1_per_class = self.val_f1_per_class(preds, target)
        for i, name in enumerate(['Wake', 'NREM', 'REM']):
            self.log(f"val/f1_{name}", f1_per_class[i], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        # store for confusion matrix callback (no extra forward pass)
        self.validation_step_outputs.append({
            'preds': preds.detach(),
            'targets': target.detach()
        })

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch.
        """
        # Provide collected outputs to callback via trainer (attribute read by callback)
        self.trainer.model_validation_outputs = self.validation_step_outputs
        self.validation_step_outputs = []

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch.
        """
        # Provide collected outputs to callback via trainer (attribute read by callback)
        self.trainer.model_training_outputs = self.training_step_outputs
        self.training_step_outputs = []

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[Dict]]: A tuple containing the optimizer and scheduler configurations.
        """
        optimizer = optimizers.__dict__[self.config.optimizer_name](
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas
        )

        if self.config.scheduler is not None:
            lr_scheduler_config = {
                "scheduler": CosineLinearWarmupLR(
                    optimizer,
                    warmup_epochs=self.config.warmup_steps,
                    max_epochs=self.config.total_training_steps
                ),
                "interval": "step",
                "monitor": None
            }

            return [optimizer], [lr_scheduler_config]

        return optimizer

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True
        )
