import datetime
import json
import os
from typing import Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from src.configs import get_config
from src.data import SleepDataset, TrainValSplitter


def create_artifacts_dir(base_path: str = "./artifacts") -> str:
    """
    Create timestamped artifacts directory.
    
    Args:
        base_path: Base path for artifacts.
        
    Returns:
        Path to created artifacts directory.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_path = os.path.join(base_path, timestamp)
    os.makedirs(artifacts_path, exist_ok=True)

    return artifacts_path


def save_fold_subjects(train_subjects: List[str], val_subjects: List[str], fold_path: str) -> None:
    """
    Save train/val subject IDs for a fold.
    
    Args:
        train_subjects: List of training subject IDs.
        val_subjects: List of validation subject IDs.
        fold_path: Path to save subjects file.
    """
    subjects_file = os.path.join(fold_path, 'subjects.json')
    
    with open(subjects_file, 'w') as f:
        json.dump({
            'train': train_subjects,
            'val': val_subjects
        }, f, indent=2)
    
    print(f"Saved subject IDs to {subjects_file}")


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute per-class accuracy (recall) metrics.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        
    Returns:
        Dictionary with per-class accuracies.
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


def save_fold_metrics(fold_idx: int, metrics: Dict, fold_path: str) -> None:
    """
    Save metrics for a single fold to text file.
    
    Args:
        fold_idx: Fold index (0-based).
        metrics: Dictionary containing metrics.
        fold_path: Path to save metrics file.
    """
    metrics_file = os.path.join(fold_path, 'metrics.txt')
    
    with open(metrics_file, 'w') as f:
        f.write(f"Fold {fold_idx + 1} - Logistic Regression Baseline\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  F1 Macro: {metrics['f1_macro']:.4f}\n\n")
        
        f.write(f"Per-class Accuracy (Recall):\n")
        for class_name in ['Wake', 'NREM', 'REM']:
            acc = metrics.get(f'{class_name}_accuracy', 0)
            samples = metrics.get(f'{class_name}_samples', 0)
            f.write(f"  {class_name}: {acc:.4f} ({samples} samples)\n")
        
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"{metrics['confusion_matrix']}\n\n")
        
        f.write(f"Classification Report:\n")
        f.write(f"{metrics['classification_report']}\n")
    
    print(f"Saved metrics to {metrics_file}")


def save_summary_metrics(all_fold_metrics: List[Dict], artifacts_path: str) -> None:
    """
    Save summary statistics across all folds.
    
    Args:
        all_fold_metrics: List of metric dictionaries from all folds (each containing 'train' and 'val' keys).
        artifacts_path: Path to artifacts directory.
    """
    summary_file = os.path.join(artifacts_path, 'kfold_summary.txt')
    
    # Extract train and val metrics separately
    train_metrics = [m['train'] for m in all_fold_metrics]
    val_metrics = [m['val'] for m in all_fold_metrics]
    
    # Compute statistics for validation set
    val_accuracies = [m['accuracy'] for m in val_metrics]
    val_f1_macros = [m['f1_macro'] for m in val_metrics]
    val_wake_accs = [m.get('Wake_accuracy', 0) for m in val_metrics]
    val_nrem_accs = [m.get('NREM_accuracy', 0) for m in val_metrics]
    val_rem_accs = [m.get('REM_accuracy', 0) for m in val_metrics]
    
    # Compute statistics for training set
    train_accuracies = [m['accuracy'] for m in train_metrics]
    train_f1_macros = [m['f1_macro'] for m in train_metrics]
    train_wake_accs = [m.get('Wake_accuracy', 0) for m in train_metrics]
    train_nrem_accs = [m.get('NREM_accuracy', 0) for m in train_metrics]
    train_rem_accs = [m.get('REM_accuracy', 0) for m in train_metrics]
    
    with open(summary_file, 'w') as f:
        f.write("K-Fold Cross-Validation Summary - Logistic Regression Baseline\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Number of folds: {len(all_fold_metrics)}\n\n")
        
        f.write("VALIDATION SET METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:  {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}\n")
        f.write(f"  F1 Macro:  {np.mean(val_f1_macros):.4f} ± {np.std(val_f1_macros):.4f}\n\n")
        
        f.write(f"Per-class Accuracy (Recall):\n")
        f.write(f"  Wake:  {np.mean(val_wake_accs):.4f} ± {np.std(val_wake_accs):.4f}\n")
        f.write(f"  NREM:  {np.mean(val_nrem_accs):.4f} ± {np.std(val_nrem_accs):.4f}\n")
        f.write(f"  REM:   {np.mean(val_rem_accs):.4f} ± {np.std(val_rem_accs):.4f}\n\n")
        
        f.write("TRAINING SET METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy:  {np.mean(train_accuracies):.4f} ± {np.std(train_accuracies):.4f}\n")
        f.write(f"  F1 Macro:  {np.mean(train_f1_macros):.4f} ± {np.std(train_f1_macros):.4f}\n\n")
        
        f.write(f"Per-class Accuracy (Recall):\n")
        f.write(f"  Wake:  {np.mean(train_wake_accs):.4f} ± {np.std(train_wake_accs):.4f}\n")
        f.write(f"  NREM:  {np.mean(train_nrem_accs):.4f} ± {np.std(train_nrem_accs):.4f}\n")
        f.write(f"  REM:   {np.mean(train_rem_accs):.4f} ± {np.std(train_rem_accs):.4f}\n\n")
        
        f.write(f"DETAILED RESULTS PER FOLD:\n")
        f.write("-" * 80 + "\n")
        for i, metrics in enumerate(all_fold_metrics):
            f.write(f"\nFold {i + 1}:\n")
            f.write(f"  Train - Accuracy: {metrics['train']['accuracy']:.4f}, F1 Macro: {metrics['train']['f1_macro']:.4f}\n")
            f.write(f"  Val   - Accuracy: {metrics['val']['accuracy']:.4f}, F1 Macro: {metrics['val']['f1_macro']:.4f}\n")
            f.write(f"  Val   - Wake: {metrics['val'].get('Wake_accuracy', 0):.4f}, ")
            f.write(f"NREM: {metrics['val'].get('NREM_accuracy', 0):.4f}, ")
            f.write(f"REM: {metrics['val'].get('REM_accuracy', 0):.4f}\n")
    
    print(f"\nSaved summary to {summary_file}")
    
    # Print to console
    print("\n" + "=" * 80)
    print("K-Fold Cross-Validation Summary - Logistic Regression Baseline")
    print("=" * 80)
    print(f"\nVALIDATION SET:")
    print(f"  Accuracy:  {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")
    print(f"  F1 Macro:  {np.mean(val_f1_macros):.4f} ± {np.std(val_f1_macros):.4f}")
    print(f"  Wake:      {np.mean(val_wake_accs):.4f} ± {np.std(val_wake_accs):.4f}")
    print(f"  NREM:      {np.mean(val_nrem_accs):.4f} ± {np.std(val_nrem_accs):.4f}")
    print(f"  REM:       {np.mean(val_rem_accs):.4f} ± {np.std(val_rem_accs):.4f}")
    print(f"\nTRAINING SET:")
    print(f"  Accuracy:  {np.mean(train_accuracies):.4f} ± {np.std(train_accuracies):.4f}")
    print(f"  F1 Macro:  {np.mean(train_f1_macros):.4f} ± {np.std(train_f1_macros):.4f}")


def train_fold(
    fold_idx: int,
    train_subjects: List[str],
    val_subjects: List[str],
    config,
    fold_path: str,
) -> Dict:
    """
    Train and evaluate logistic regression for a single fold.
    
    Args:
        fold_idx: Fold index (0-based).
        train_subjects: List of training subject IDs.
        val_subjects: List of validation subject IDs.
        config: Configuration object.
        fold_path: Path to save fold artifacts.
        
    Returns:
        Dictionary containing fold metrics.
    """
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}")
    print(f"{'='*60}")
    print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")
    
    # Save subject IDs for this fold
    save_fold_subjects(train_subjects, val_subjects, fold_path)
    
    # Load datasets
    print("Loading training data...")
    train_dataset = SleepDataset(
        main_data_path=config.main_data_path,
        subjects_id=train_subjects,
        lookback_window=config.lookback_window,
        hr_window_sizes=config.hr_window_sizes,
        motion_gravity_constant=config.motion_gravity_constant,
        normalization_epsilon=config.normalization_epsilon,
        psg_buffer=config.psg_buffer,
        min_samples_for_std=config.min_samples_for_std,
    )
    
    print("Loading validation data...")
    val_dataset = SleepDataset(
        main_data_path=config.main_data_path,
        subjects_id=val_subjects,
        lookback_window=config.lookback_window,
        hr_window_sizes=config.hr_window_sizes,
        motion_gravity_constant=config.motion_gravity_constant,
        normalization_epsilon=config.normalization_epsilon,
        psg_buffer=config.psg_buffer,
        min_samples_for_std=config.min_samples_for_std,
    )
    
    # Flatten features for logistic regression
    X_train = train_dataset.data.reshape(len(train_dataset.data), -1)
    y_train = train_dataset.labels
    
    X_val = val_dataset.data.reshape(len(val_dataset.data), -1)
    y_val = val_dataset.labels
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    print(f"Feature dimensions: {X_train.shape}")
    
    # Get hyperparameters from config or use defaults
    max_iter = getattr(config, 'logreg_max_iter', 3000)
    class_weight = getattr(config, 'logreg_class_weight', 'balanced')
    random_state = getattr(config, 'logreg_random_state', 42)
    
    # Train logistic regression
    print(f"Training Logistic Regression (max_iter={max_iter}, class_weight={class_weight})...")
    classifier = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
        verbose=0
    )
    
    classifier.fit(X_train, y_train)
    
    # Evaluate on train set
    print("Evaluating on train set...")
    y_train_pred = classifier.predict(X_train)
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'f1_macro': f1_score(y_train, y_train_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_train, y_train_pred),
        'classification_report': classification_report(y_train, y_train_pred, target_names=['Wake', 'NREM', 'REM'])
    }
    # Add per-class metrics
    train_metrics.update(compute_per_class_metrics(y_train, y_train_pred))
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    y_val_pred = classifier.predict(X_val)
    val_metrics = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'f1_macro': f1_score(y_val, y_val_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_val, y_val_pred),
        'classification_report': classification_report(y_val, y_val_pred, target_names=['Wake', 'NREM', 'REM'])
    }
    # Add per-class metrics
    val_metrics.update(compute_per_class_metrics(y_val, y_val_pred))
    
    # Print results
    print(f"\nFold {fold_idx + 1} Results:")
    print(f"  Train - Accuracy: {train_metrics['accuracy']:.4f}, F1 Macro: {train_metrics['f1_macro']:.4f}")
    print(f"  Val   - Accuracy: {val_metrics['accuracy']:.4f}, F1 Macro: {val_metrics['f1_macro']:.4f}")
    print(f"  Val Wake Acc: {val_metrics.get('Wake_accuracy', 0):.4f}")
    print(f"  Val NREM Acc: {val_metrics.get('NREM_accuracy', 0):.4f}")
    print(f"  Val REM Acc:  {val_metrics.get('REM_accuracy', 0):.4f}")
    
    # Save metrics to files
    save_fold_metrics(fold_idx, train_metrics, fold_path)
    
    val_fold_path = os.path.join(fold_path, 'val_metrics.txt')
    with open(val_fold_path, 'w') as f:
        f.write(f"Fold {fold_idx + 1} - Validation Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy: {val_metrics['accuracy']:.4f}\n")
        f.write(f"  F1 Macro: {val_metrics['f1_macro']:.4f}\n\n")
        f.write(f"Per-class Accuracy (Recall):\n")
        for class_name in ['Wake', 'NREM', 'REM']:
            acc = val_metrics.get(f'{class_name}_accuracy', 0)
            samples = val_metrics.get(f'{class_name}_samples', 0)
            f.write(f"  {class_name}: {acc:.4f} ({samples} samples)\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"{val_metrics['confusion_matrix']}\n\n")
        f.write(f"Classification Report:\n")
        f.write(f"{val_metrics['classification_report']}\n")
    
    # Save model
    import joblib
    model_path = os.path.join(fold_path, f'model_fold_{fold_idx}.pkl')
    joblib.dump(classifier, model_path)
    print(f"Saved model to {model_path}")
    
    return {
        'train': train_metrics,
        'val': val_metrics
    }


def run_kfold_baseline(
    n_folds: int = None,
    val_split: float = None,
    random_seeds: List[int] = None,
) -> None:
    """
    Run k-fold cross-validation with logistic regression baseline.
    
    Args:
        n_folds: Number of folds for cross-validation. If None, uses config.n_folds.
        val_split: Validation split ratio. If None, uses config.val_split.
        random_seeds: List of random seeds for each fold. If None, uses config.random_seeds.
    """
    print("="*80)
    print("Logistic Regression K-Fold Baseline")
    print("="*80)
    
    # Load configuration
    config = get_config()
    
    # Use config parameters if not provided
    if n_folds is None:
        n_folds = config.n_folds
    if val_split is None:
        val_split = config.val_split
    if random_seeds is None:
        random_seeds = config.random_seeds[:n_folds]
    
    # Get hyperparameters from config or use defaults
    max_iter = getattr(config, 'logreg_max_iter', 3000)
    class_weight = getattr(config, 'logreg_class_weight', 'balanced')
    random_state = getattr(config, 'logreg_random_state', 42)
    
    # Create artifacts directory
    artifacts_path = create_artifacts_dir(config.artifacts_path)
    print(f"\nArtifacts will be saved to: {artifacts_path}")
    
    print(f"\nRunning {n_folds}-fold cross-validation")
    print(f"Validation split: {val_split:.1%}")
    print(f"Random seeds: {random_seeds}")
    print(f"\nLogistic Regression Hyperparameters:")
    print(f"  max_iter: {max_iter}")
    print(f"  class_weight: {class_weight}")
    print(f"  random_state: {random_state}")
    
    # Save configuration
    config_file = os.path.join(artifacts_path, 'config.txt')
    with open(config_file, 'w') as f:
        f.write("Logistic Regression Baseline Configuration\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Number of folds: {n_folds}\n")
        f.write(f"Validation split: {val_split}\n")
        f.write(f"Random seeds: {random_seeds}\n\n")
        f.write(f"Model parameters:\n")
        f.write(f"  max_iter: {max_iter}\n")
        f.write(f"  class_weight: {class_weight}\n")
        f.write(f"  random_state: {random_state}\n\n")
        f.write(f"Data configuration:\n")
        f.write(f"  lookback_window: {config.lookback_window}\n")
        f.write(f"  hr_window_sizes: {config.hr_window_sizes}\n")
        f.write(f"  motion_gravity_constant: {config.motion_gravity_constant}\n")
    
    # Run k-fold cross-validation
    all_fold_metrics = []
    
    for fold_idx in tqdm(range(n_folds), desc="Folds"):
        # Create fold directory
        fold_path = os.path.join(artifacts_path, f'fold_{fold_idx}')
        os.makedirs(fold_path, exist_ok=True)
        
        # Split data
        train_subjects, val_subjects = TrainValSplitter(
            main_data_path=config.main_data_path,
            val_split=val_split,
            random_seed=random_seeds[fold_idx]
        ).split()
        
        # Train and evaluate fold
        fold_metrics = train_fold(
            fold_idx=fold_idx,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            config=config,
            fold_path=fold_path,
        )
        
        all_fold_metrics.append(fold_metrics)
    
    # Save summary
    save_summary_metrics(all_fold_metrics, artifacts_path)

if __name__ == "__main__":
    run_kfold_baseline()