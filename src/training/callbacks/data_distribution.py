from collections import Counter
from typing import List, Optional

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch.callbacks import Callback


class ClassDistributionLogger(Callback):
    """Callback to log class distribution at the start of training."""
    
    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        """
        Initialize the ClassDistributionLogger.
        
        Args:
            class_names: Names of classes for labeling (optional).
        """
        super().__init__()
        self.class_names = class_names or ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]
        self.logged = False
    
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Log class distribution once at the beginning of training.
        
        Args:
            trainer: The PyTorch Lightning trainer.
            pl_module: The LightningModule being trained.
        """
        if self.logged:
            return
        
        all_labels = []
        train_loader = trainer.train_dataloader
        
        print("Computing class distribution from training data...")
        for _, labels in train_loader.dataset:
            labels_flat = labels.numpy().flatten()
            labels_flat = labels_flat[labels_flat != -100]
            all_labels.extend(labels_flat)
        
        all_labels = np.array(all_labels)
        class_counts = Counter(all_labels)
        total = len(all_labels)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        classes = list(range(len(self.class_names)))
        counts = [class_counts.get(i, 0) for i in classes]
        
        ax1.bar(classes, counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Class Distribution (Counts)')
        ax1.set_xticks(classes)
        ax1.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        for i, (cls, count) in enumerate(zip(classes, counts)):
            ax1.text(cls, count, f'{count}', ha='center', va='bottom')
        
        percentages = [100 * count / total for count in counts]
        
        ax2.bar(classes, percentages, color='coral', alpha=0.7)
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Class Distribution (Percentages)')
        ax2.set_xticks(classes)
        ax2.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        for _, (cls, pct) in enumerate(zip(classes, percentages)):
            ax2.text(cls, pct, f'{pct:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if trainer.logger is not None:
            trainer.logger.experiment.add_figure('train/class_distribution', fig, global_step=0)
        
        plt.close(fig)
        self.logged = True
