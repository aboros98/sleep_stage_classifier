from typing import List, Optional

import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


class ModelClassWeightsLogger(Callback):
    """Callback to extract and log class weights from the model's loss function."""
    
    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        """
        Initialize the ModelClassWeightsLogger.
        
        Args:
            class_names: Names of classes for labeling (optional).
        """
        super().__init__()
        self.class_names = class_names or ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]
        self.logged = False
    
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Extract and log class weights from the model at the beginning of training.
        
        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being trained.
        """
        if self.logged:
            return
        
        class_weights = None
        
        if hasattr(pl_module.model, 'loss_fn'):
            loss_fn = pl_module.model.loss_fn
            
            if hasattr(loss_fn, 'alpha') and loss_fn.alpha is not None:
                class_weights = loss_fn.alpha.cpu().numpy()
            elif hasattr(loss_fn, 'weight') and loss_fn.weight is not None:
                class_weights = loss_fn.weight.cpu().numpy()
        
        if class_weights is None:
            print("No class weights found in model. Loss function may not use weighted classes.")
            self.logged = True
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = list(range(len(class_weights)))
        
        ax.bar(classes, class_weights, color='purple', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_title('Model Class Weights', fontsize=14)
        ax.set_xticks(classes)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Baseline (1.0)')
        ax.legend()
        
        for _, (cls, weight) in enumerate(zip(classes, class_weights)):
            ax.text(cls, weight, f'{weight:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if trainer.logger is not None:
            trainer.logger.experiment.add_figure('model/class_weights', fig, global_step=0)
        
        plt.close(fig)
        self.logged = True
