import datetime
import os
from typing import Dict, List

import lightning as L
import numpy as np
import torch

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from ml_collections import ConfigDict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .callbacks import ClassDistributionLogger, ConfusionMatrixLogger, GradNormLogger, ModelClassWeightsLogger
from .lightning_module import TrainingModel


class SimpleTrainer:
    """Trainer class for single train/val split training."""
    
    def __init__(self, config: ConfigDict) -> None:
        """
        Initialize the SimpleTrainer.
        
        Args:
            config: Configuration dictionary with all training parameters.
        """
        self.config = config

        self.artifacts_path = os.path.join(
            self.config.artifacts_path,
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        os.makedirs(self.artifacts_path, exist_ok=True)

    def _init_callbacks(self) -> List:
        """
        Initialize training callbacks.
            
        Returns:
            List of initialized callback objects.
        """
        lr_monitor = LearningRateMonitor(logging_interval="step")
        
        grad_monitor = GradNormLogger(
            log_all_layers_norm=False,
            norm_type=self.config.grad_norm_type,
            log_on_step=True,
            log_on_epoch=False
        )

        checkpoint_callback = ModelCheckpoint(
            monitor=self.config.checkpoint_monitor,
            mode='max',
            save_last=True,
            save_top_k=self.config.checkpoint_save_top_k,
            save_weights_only=True,
            every_n_epochs=self.config.checkpoint_every_n_epochs,
            auto_insert_metric_name=True,
            dirpath=self.artifacts_path
        )

        early_stopping = EarlyStopping(
            monitor=self.config.early_stopping_monitor,
            patience=self.config.early_stopping_patience,
            mode=self.config.early_stopping_mode,
            verbose=True
        )

        confusion_matrix_logger = ConfusionMatrixLogger(
            num_classes=self.config.num_classes,
            class_names=['Wake', 'NREM', 'REM'],
            ignore_index=self.config.ignore_index,
            log_every_n_epochs=self.config.confusion_matrix_log_every_n_epochs,
            log_train=self.config.confusion_matrix_log_train,
            log_val=self.config.confusion_matrix_log_val
        )

        class_distribution_logger = ClassDistributionLogger(class_names=['Wake', 'NREM', 'REM'])
        model_class_weights_logger = ModelClassWeightsLogger(class_names=['Wake', 'NREM', 'REM'])

        return [
            lr_monitor,
            grad_monitor,
            checkpoint_callback,
            early_stopping,
            confusion_matrix_logger,
            class_distribution_logger,
            model_class_weights_logger
        ]

    def _compute_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Compute per-class accuracy (recall) and sample counts.
        
        Args:
            y_true: True labels array.
            y_pred: Predicted labels array.
            
        Returns:
            Dictionary containing per-class accuracy and sample counts.
        """
        class_names = ['Wake', 'NREM', 'REM']
        metrics = {}
        
        for class_idx, class_name in enumerate(class_names):
            class_mask = y_true == class_idx
            if class_mask.sum() > 0:
                class_acc = accuracy_score(y_true[class_mask], y_pred[class_mask])
                metrics[f'{class_name}_accuracy'] = class_acc
                metrics[f'{class_name}_samples'] = int(class_mask.sum())
            else:
                metrics[f'{class_name}_accuracy'] = 0.0
                metrics[f'{class_name}_samples'] = 0
        
        return metrics

    def _evaluate(self, model: TrainingModel, dataloader) -> Dict:
        """
        Evaluate model on a dataloader and return metrics.
        
        Args:
            model: The trained model to evaluate.
            dataloader: DataLoader containing evaluation data.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        model.eval()
        model.to(self.config.device_list[0])
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                source, target = batch
                source = source.to(model.device)
                target = target.to(model.device)
                
                logits, _ = model(source, target)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        y_pred = np.concatenate(all_preds).flatten()
        y_true = np.concatenate(all_targets).flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true,
                y_pred,
                target_names=['Wake', 'NREM', 'REM'],
                zero_division=0
            )
        }
        
        per_class_metrics = self._compute_per_class_metrics(y_true, y_pred)
        metrics.update(per_class_metrics)
        
        return metrics

    def _print_results(self, metrics: Dict) -> None:
        """
        Print detailed results.
        
        Args:
            metrics: Dictionary containing evaluation metrics.
        """
        print(f"\n{'='*60}")
        print(f"Training Results:")
        print(f"Overall Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1 Macro: {metrics['f1_macro']:.3f}")
        
        print(f"\nPer-class Accuracy (Recall):")
        for class_name in ['Wake', 'NREM', 'REM']:
            acc = metrics.get(f'{class_name}_accuracy', 0)
            samples = metrics.get(f'{class_name}_samples', 0)
            print(f"  {class_name}: {acc:.3f} ({samples} samples)")
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print(f"\n{metrics['classification_report']}")

    def run(self) -> None:
        """Run the training process with a single train/val split."""
        L.seed_everything(self.config.seed)

        callbacks = self._init_callbacks()
        model = TrainingModel(config=self.config, artifacts_path=self.artifacts_path)

        trainer = L.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator=self.config.accelerator,
            devices=self.config.device_list,
            precision=self.config.precision,
            deterministic=self.config.deterministic,
            logger=True,
            log_every_n_steps=self.config.log_every_n_steps,
            check_val_every_n_epoch=self.config.check_val_every_n_epoch,
            callbacks=callbacks,
            default_root_dir=self.artifacts_path
        )

        trainer.fit(model)

        # Evaluate on validation set
        val_metrics = self._evaluate(model, model.val_dataloader())
        self._print_results(val_metrics)
        
        # Save metrics to file
        metrics_path = os.path.join(self.artifacts_path, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Training Results:\n")
            f.write(f"Overall Accuracy: {val_metrics['accuracy']:.3f}\n")
            f.write(f"F1 Macro: {val_metrics['f1_macro']:.3f}\n\n")
            f.write(f"Per-class Accuracy (Recall):\n")
            for class_name in ['Wake', 'NREM', 'REM']:
                acc = val_metrics.get(f'{class_name}_accuracy', 0)
                samples = val_metrics.get(f'{class_name}_samples', 0)
                f.write(f"  {class_name}: {acc:.3f} ({samples} samples)\n")
            f.write(f"\nConfusion Matrix:\n{val_metrics['confusion_matrix']}\n")
            f.write(f"\n{val_metrics['classification_report']}")
