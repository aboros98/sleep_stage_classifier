from typing import List, Optional

import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from lightning.pytorch.callbacks import Callback
from torchmetrics import ConfusionMatrix


class ConfusionMatrixLogger(Callback):
    """Callback to log confusion matrices during training and validation."""
    
    def __init__(
        self,
        num_classes: int = 5,
        class_names: Optional[List[str]] = None,
        ignore_index: int = -100,
        log_every_n_epochs: int = 1,
        log_train: bool = True,
        log_val: bool = True
    ) -> None:
        """
        Initialize the ConfusionMatrixLogger.
        
        Args:
            num_classes: Number of classification classes.
            class_names: Names of classes for labeling (optional).
            ignore_index: Index to ignore in confusion matrix computation.
            log_every_n_epochs: Frequency of logging (every N epochs).
            log_train: Whether to log training confusion matrix.
            log_val: Whether to log validation confusion matrix.
        """
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        self.ignore_index = ignore_index
        self.log_every_n_epochs = log_every_n_epochs
        self.log_train = log_train
        self.log_val = log_val
        
        self.train_confusion_matrix = ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index
        )
        self.val_confusion_matrix = ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index
        )

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Compute and log confusion matrix at the end of each training epoch.
        
        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being trained.
        """
        if not self.log_train:
            return
            
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        outputs = getattr(trainer, 'model_training_outputs', None)
        if not outputs:
            return
        
        all_preds = torch.cat([x['preds'] for x in outputs]).view(-1)
        all_targets = torch.cat([x['targets'] for x in outputs]).view(-1)
        
        valid_mask = all_targets != self.ignore_index
        all_preds = all_preds[valid_mask]
        all_targets = all_targets[valid_mask]
        
        self.train_confusion_matrix = self.train_confusion_matrix.to(all_preds.device)
        cm = self.train_confusion_matrix(all_preds, all_targets)
        
        fig = self._create_confusion_matrix_figure(cm.cpu().numpy(), trainer.current_epoch, 'Train')
        
        if trainer.logger is not None:
            trainer.logger.experiment.add_figure('train/confusion_matrix', fig, global_step=trainer.current_epoch)
        
        plt.close(fig)
        self.train_confusion_matrix.reset()

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Compute and log confusion matrix at the end of each validation epoch.
        
        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being trained.
        """
        if not self.log_val:
            return
            
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        outputs = getattr(trainer, 'model_validation_outputs', None)
        if not outputs:
            return
        
        all_preds = torch.cat([x['preds'] for x in outputs]).view(-1)
        all_targets = torch.cat([x['targets'] for x in outputs]).view(-1)
        
        valid_mask = all_targets != self.ignore_index
        all_preds = all_preds[valid_mask]
        all_targets = all_targets[valid_mask]
        
        self.val_confusion_matrix = self.val_confusion_matrix.to(all_preds.device)
        cm = self.val_confusion_matrix(all_preds, all_targets)
        
        fig = self._create_confusion_matrix_figure(cm.cpu().numpy(), trainer.current_epoch, 'Validation')
        
        if trainer.logger is not None:
            trainer.logger.experiment.add_figure('val/confusion_matrix', fig, global_step=trainer.current_epoch)
        
        plt.close(fig)
        self.val_confusion_matrix.reset()

    def _create_confusion_matrix_figure(
        self,
        cm: torch.Tensor,
        epoch: int,
        split: str = 'Validation'
    ) -> plt.Figure:
        """
        Create a matplotlib figure for the confusion matrix.
        
        Args:
            cm: Confusion matrix as a numpy array.
            epoch: Current epoch number.
            split: Dataset split name ('Train' or 'Validation').
            
        Returns:
            Matplotlib figure object with the confusion matrix heatmap.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2%',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
            cbar_kws={'label': 'Proportion'},
            vmin=0,
            vmax=1
        )
        
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title(f'{split} Confusion Matrix - Epoch {epoch}', fontsize=14)
        
        plt.tight_layout()

        return fig
