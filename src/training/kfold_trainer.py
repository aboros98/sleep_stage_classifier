import datetime
import os
from typing import Dict, List, Tuple

import lightning as L
import numpy as np
import torch

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from ml_collections import ConfigDict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data import TrainValSplitter

from .callbacks import ClassDistributionLogger, ConfusionMatrixLogger, GradNormLogger, ModelClassWeightsLogger
from .lightning_module import TrainingModel


class KFoldTrainer:
    """Trainer class for K-fold cross-validation training."""
    
    def __init__(self, config: ConfigDict) -> None:
        """
        Initialize the KFoldTrainer.
        
        Args:
            config: Configuration dictionary with all training parameters.
        """
        self.config = config

        self.artifacts_path = os.path.join(
            self.config.artifacts_path,
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        os.makedirs(self.artifacts_path, exist_ok=True)

    def _init_callbacks(self, fold_artifacts_path: str) -> List:
        """
        Initialize training callbacks.
        
        Args:
            fold_artifacts_path: Path to save fold-specific artifacts.
            
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
            dirpath=fold_artifacts_path
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

    def _evaluate_fold(self, model: TrainingModel, dataloader) -> Dict:
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
        
        # Add per-class F1 scores
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        for idx, class_name in enumerate(['Wake', 'NREM', 'REM']):
            metrics[f'f1_{class_name}'] = f1_per_class[idx]
        
        per_class_metrics = self._compute_per_class_metrics(y_true, y_pred)
        metrics.update(per_class_metrics)
        
        return metrics

    def _print_fold_results(self, fold: int, metrics: Dict) -> None:
        """
        Print detailed results for a single fold.
        
        Args:
            fold: Fold index (0-based).
            metrics: Dictionary containing fold metrics.
        """
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1} Results:")
        print(f"Overall Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1 Macro: {metrics['f1_macro']:.3f}")
        
        print(f"\nPer-class F1 Scores:")
        for class_name in ['Wake', 'NREM', 'REM']:
            f1 = metrics.get(f'f1_{class_name}', 0)
            print(f"  {class_name}: {f1:.3f}")
        
        print(f"\nPer-class Accuracy (Recall):")
        for class_name in ['Wake', 'NREM', 'REM']:
            acc = metrics.get(f'{class_name}_accuracy', 0)
            samples = metrics.get(f'{class_name}_samples', 0)
            print(f"  {class_name}: {acc:.3f} ({samples} samples)")
        
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print(f"\n{metrics['classification_report']}")

    def _compute_summary_statistics(self, all_fold_metrics: List[Dict]) -> Dict:
        """
        Compute summary statistics across all folds.
        
        Args:
            all_fold_metrics: List of metric dictionaries from all folds.
            
        Returns:
            Dictionary containing mean and std for each metric.
        """
        summary = {}
        
        # Overall metrics
        accuracies = [m['accuracy'] for m in all_fold_metrics]
        f1_macros = [m['f1_macro'] for m in all_fold_metrics]
        
        summary['val_accuracy'] = {'mean': np.mean(accuracies), 'std': np.std(accuracies)}
        summary['val_f1_macro'] = {'mean': np.mean(f1_macros), 'std': np.std(f1_macros)}
        
        # Per-class F1 scores
        for class_name in ['Wake', 'NREM', 'REM']:
            f1_scores = [m.get(f'f1_{class_name}', 0) for m in all_fold_metrics]
            summary[f'val_f1_{class_name}'] = {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)}
        
        # Per-class accuracy
        for class_name in ['Wake', 'NREM', 'REM']:
            class_accs = [m.get(f'{class_name}_accuracy', 0) for m in all_fold_metrics]
            summary[f'val_{class_name}_accuracy'] = {'mean': np.mean(class_accs), 'std': np.std(class_accs)}
        
        return summary

    def _print_summary_results(self, summary: Dict) -> None:
        """
        Print summary statistics across all folds.
        
        Args:
            summary: Dictionary containing summary statistics.
        """
        print(f"\n{'='*60}")
        print(f"K-Fold Cross-Validation Summary")
        print(f"{'='*60}")
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {summary['val_accuracy']['mean']:.3f} ± {summary['val_accuracy']['std']:.3f}")
        print(f"  F1 Macro: {summary['val_f1_macro']['mean']:.3f} ± {summary['val_f1_macro']['std']:.3f}")
        
        print(f"\nPer-class F1 Scores:")
        for class_name in ['Wake', 'NREM', 'REM']:
            mean = summary[f'val_f1_{class_name}']['mean']
            std = summary[f'val_f1_{class_name}']['std']
            print(f"  {class_name}: {mean:.3f} ± {std:.3f}")
        
        print(f"\nPer-class Accuracy (Recall):")
        for class_name in ['Wake', 'NREM', 'REM']:
            mean = summary[f'val_{class_name}_accuracy']['mean']
            std = summary[f'val_{class_name}_accuracy']['std']
            print(f"  {class_name}: {mean:.3f} ± {std:.3f}")

    def _save_summary_to_file(self, summary: Dict, n_splits: int) -> None:
        """
        Save summary statistics to a file.
        
        Args:
            summary: Dictionary containing summary statistics.
            n_splits: Number of folds used.
        """
        summary_path = os.path.join(self.artifacts_path, 'kfold_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"K-Fold Cross-Validation Summary ({n_splits} folds)\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"Overall Metrics:\n")
            f.write(f"  Accuracy: {summary['val_accuracy']['mean']:.3f} ± {summary['val_accuracy']['std']:.3f}\n")
            f.write(f"  F1 Macro: {summary['val_f1_macro']['mean']:.3f} ± {summary['val_f1_macro']['std']:.3f}\n\n")
            
            f.write(f"Per-class F1 Scores:\n")
            for class_name in ['Wake', 'NREM', 'REM']:
                mean = summary[f'val_f1_{class_name}']['mean']
                std = summary[f'val_f1_{class_name}']['std']
                f.write(f"  {class_name}: {mean:.3f} ± {std:.3f}\n")
            
            f.write(f"\nPer-class Accuracy (Recall):\n")
            for class_name in ['Wake', 'NREM', 'REM']:
                mean = summary[f'val_{class_name}_accuracy']['mean']
                std = summary[f'val_{class_name}_accuracy']['std']
                f.write(f"  {class_name}: {mean:.3f} ± {std:.3f}\n")

    def run_kfold(self, n_splits: int = 5) -> Tuple[List[Dict], Dict]:
        """
        Run K-fold cross-validation training.
        
        Args:
            n_splits: Number of folds for cross-validation.
            
        Returns:
            Tuple containing:
                - List of metric dictionaries from all folds
                - Summary dictionary with mean and std for each metric
        """
        all_fold_metrics = []

        for i in range(n_splits):
            print(f"\n{'='*60}")
            print(f"Training Fold {i + 1}/{n_splits}")
            print(f"{'='*60}")

            train_subjects, val_subjects = TrainValSplitter(
                main_data_path=self.config.main_data_path,
                val_split=self.config.val_split,
                random_seed=self.config.random_seeds[i]
            ).split()
            
            print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")
            
            fold_artifacts_path = os.path.join(self.artifacts_path, f'fold_{i + 1}')
            os.makedirs(fold_artifacts_path, exist_ok=True)
            
            callbacks = self._init_callbacks(fold_artifacts_path)

            model = TrainingModel(
                config=self.config,
                artifacts_path=fold_artifacts_path,
                train_subjects_id=train_subjects,
                val_subjects_id=val_subjects
            )
            
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
                default_root_dir=fold_artifacts_path
            )
            
            trainer.fit(model)
            
            val_metrics = self._evaluate_fold(model, model.val_dataloader())
            all_fold_metrics.append(val_metrics)
            
            self._print_fold_results(i, val_metrics)
            
            # Save fold metrics to file
            fold_metrics_path = os.path.join(fold_artifacts_path, 'metrics.txt')
            with open(fold_metrics_path, 'w') as f:
                f.write(f"Fold {i + 1} Results:\n")
                f.write(f"Overall Accuracy: {val_metrics['accuracy']:.3f}\n")
                f.write(f"F1 Macro: {val_metrics['f1_macro']:.3f}\n\n")
                f.write(f"Per-class F1 Scores:\n")
                for class_name in ['Wake', 'NREM', 'REM']:
                    f1 = val_metrics.get(f'f1_{class_name}', 0)
                    f.write(f"  {class_name}: {f1:.3f}\n")
                f.write(f"\nPer-class Accuracy (Recall):\n")
                for class_name in ['Wake', 'NREM', 'REM']:
                    acc = val_metrics.get(f'{class_name}_accuracy', 0)
                    samples = val_metrics.get(f'{class_name}_samples', 0)
                    f.write(f"  {class_name}: {acc:.3f} ({samples} samples)\n")
                f.write(f"\nConfusion Matrix:\n{val_metrics['confusion_matrix']}\n")
                f.write(f"\n{val_metrics['classification_report']}")
        
        # Compute and print summary
        summary = self._compute_summary_statistics(all_fold_metrics)
        self._print_summary_results(summary)
        self._save_summary_to_file(summary, n_splits)
        
        return all_fold_metrics, summary
