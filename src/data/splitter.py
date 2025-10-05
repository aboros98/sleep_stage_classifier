import glob
import os
from typing import List, Tuple

import numpy as np


class TrainValSplitter:
    """Splits dataset into training and validation sets based on subject IDs."""
    
    def __init__(
        self,
        main_data_path: str,
        val_split: float = 0.1,
        random_seed: int = 42
    ) -> None:
        """
        Initialize the TrainValSplitter.
        
        Args:
            main_data_path: Path to the main data directory containing subject files.
            val_split: Proportion of subjects to use for validation (0.0 to 1.0).
            random_seed: Random seed for reproducible splits.
        """
        self.val_split = val_split
        self.random_seed = random_seed
        
        files = glob.glob(os.path.join(main_data_path, "labels/*.txt"))
        self.subjects_id = [os.path.basename(f).split('_')[0] for f in files]

    def split(self) -> Tuple[List[str], List[str]]:
        """
        Split the dataset into training and validation sets based on subject IDs.
        
        Returns:
            Tuple containing:
                - train_indices: List of subject IDs for training.
                - val_indices: List of subject IDs for validation.
        """
        num_samples = len(self.subjects_id)
        split = int(self.val_split * num_samples)

        np.random.seed(self.random_seed)
        np.random.shuffle(self.subjects_id)

        train_indices, val_indices = self.subjects_id[split:], self.subjects_id[:split]

        return train_indices, val_indices
