from src.configs import get_config
from src.training.kfold_trainer import KFoldTrainer
import os

if __name__ == "__main__":
    config = get_config()
    
    # Get all subject IDs
    labels_dir = os.path.join(config.main_data_path, 'labels')
    all_subjects = [f.replace('_labeled_sleep.txt', '') 
                   for f in os.listdir(labels_dir) 
                   if f.endswith('_labeled_sleep.txt')]
    all_subjects = sorted(all_subjects)
    
    print(f"Found {len(all_subjects)} subjects")
    
    # Run k-fold cross-validation
    trainer = KFoldTrainer(config)
    fold_results, summary = trainer.run_kfold(n_splits=5)
    
    # Print final summary
    print(f"\nFINAL RESULTS:")
    print(f"Val F1 Macro: {summary['val_f1_macro']['mean']:.3f} ± {summary['val_f1_macro']['std']:.3f}")
    print(f"REM F1: {summary['val_f1_REM']['mean']:.3f} ± {summary['val_f1_REM']['std']:.3f}")