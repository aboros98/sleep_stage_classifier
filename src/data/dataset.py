import os
from multiprocessing import Pool, cpu_count
from typing import List, Tuple

import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from src.utils.io_utils import read_txt_file
from src.utils.transform_utils import compute_rolling_std


def resample_data(data: np.ndarray, target_length: int) -> np.ndarray:
    """
    Resample time series data to a target length using linear interpolation.
    
    Args:
        data: Input time series data array.
        target_length: Desired length of the resampled data.
        
    Returns:
        Resampled data array with shape matching target_length.
    """
    old_time = np.linspace(0, 1, len(data))
    new_time = np.linspace(0, 1, target_length)
    f = interp1d(old_time, data, kind='linear', axis=0)
    
    return f(new_time)


def preprocess_for_inference(
    hr_data: np.ndarray,
    motion_data: np.ndarray,
    target_duration_seconds: int,
    hr_window_sizes: List[int] = None,
    motion_gravity_constant: float = 1.0,
    normalization_epsilon: float = 1e-8,
    min_samples_for_std: int = 2,
) -> np.ndarray:
    """
    Simplified preprocessing for inference when you have raw HR and motion data.
    
    This function assumes you already have:
    - Filtered heart rate data (BPM values)
    - Filtered 3D accelerometer data
    - Known duration of the measurement
    
    Args:
        hr_data: Heart rate measurements (BPM), any sampling rate.
        motion_data: 3D accelerometer measurements, shape (n_samples, 3).
        target_duration_seconds: Total duration to resample to (in seconds, at 1 Hz).
        lookback_window: Number of seconds to look back for each prediction.
        hr_window_sizes: List of window sizes for HR rolling std computation.
        motion_gravity_constant: Gravity constant for motion magnitude computation.
        normalization_epsilon: Epsilon for numerical stability in normalization.
        min_samples_for_std: Minimum samples required to compute rolling std.
        
    Returns:
        features: Array of shape (target_duration_seconds, n_features)
                  where n_features = 1 (motion) + len(hr_window_sizes) (HR std)
    """
    if hr_window_sizes is None:
        hr_window_sizes = [30, 60, 120]
    
    # Compute motion magnitude
    motion_magnitude = ((motion_data ** 2).sum(-1) ** 0.5 - motion_gravity_constant).clip(0)
    
    # Resample data to 1 Hz (target_duration_seconds)
    motion_data_resampled = resample_data(motion_magnitude, target_duration_seconds)
    hr_data_resampled = resample_data(hr_data, target_duration_seconds)
    
    # Normalize heart rate data (subject-specific)
    subject_hr_mean = hr_data_resampled.mean()
    subject_hr_std = hr_data_resampled.std()
    hr_normalized = (hr_data_resampled - subject_hr_mean) / (subject_hr_std + normalization_epsilon)
    
    # Compute rolling standard deviations for different window sizes
    hr_data_std_devs = np.stack(
        [compute_rolling_std(hr_normalized, window_size, min_samples_for_std) for window_size in hr_window_sizes],
        axis=-1
    )
    
    # Stack motion and heart rate features
    features = np.concatenate([motion_data_resampled.reshape(-1, 1), hr_data_std_devs], axis=-1)
    
    return features


class SleepDataset(Dataset):
    """
    Dataset for sleep stage classification from heart rate and motion data.
    
    Processes multi-subject physiological data with sliding windows for temporal features.
    """
    
    def __init__(
        self,
        main_data_path: str,
        subjects_id: List[str],
        lookback_window: int = 512,
        artifacts_path: str = "./artifacts",
        n_jobs: int = -1,
        hr_window_sizes: List[int] = None,
        motion_gravity_constant: float = 1.0,
        normalization_epsilon: float = 1e-8,
        unknown_label: int = -1,
        label_merge_value: int = 1,
        label_rem_value: int = 2,
        psg_buffer: int = 30,
        min_samples_for_std: int = 2,
    ) -> None:
        """
        Initialize the SleepDataset.
        
        Args:
            main_data_path: Path to the main data directory containing subject data.
            subjects_id: List of subject IDs to include in the dataset.
            lookback_window: Number of seconds to look back for each prediction.
            artifacts_path: Path to save/load artifacts and scalers.
            n_jobs: Number of parallel jobs for data processing (-1 uses all CPUs).
            hr_window_sizes: List of window sizes for HR rolling std computation.
            motion_gravity_constant: Gravity constant for motion magnitude computation.
            normalization_epsilon: Epsilon for numerical stability in normalization.
            unknown_label: Label value representing unknown/invalid labels.
            label_merge_value: Value to merge N3/N4 labels into (NREM).
            label_rem_value: Value to remap REM labels to.
            psg_buffer: Buffer time in seconds after PSG end.
            min_samples_for_std: Minimum samples required to compute rolling std.
        """
        self.main_data_path = main_data_path
        self.artifacts_path = artifacts_path
        self.lookback_window = lookback_window
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.hr_window_sizes = hr_window_sizes if hr_window_sizes is not None else [30, 60, 120]
        self.motion_gravity_constant = motion_gravity_constant
        self.normalization_epsilon = normalization_epsilon
        self.unknown_label = unknown_label
        self.label_merge_value = label_merge_value
        self.label_rem_value = label_rem_value
        self.psg_buffer = psg_buffer
        self.min_samples_for_std = min_samples_for_std

        os.makedirs(self.artifacts_path, exist_ok=True)
        self.data, self.labels = self._build_dataset(subjects_id)
        self.n_features = self.data.shape[-1]
        self.labels = self.labels.astype(np.int64)

    def _build_dataset(self, subjects_id: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the complete dataset by processing all subjects in parallel.
        
        Args:
            subjects_id: List of subject IDs to process.
            
        Returns:
            Tuple containing concatenated features and labels arrays.
        """
        with Pool(processes=self.n_jobs) as pool:
            results = pool.map(self._process_single_subject, subjects_id)
        
        valid_results = [(feats, labs) for feats, labs in results if len(feats) > 0 and len(labs) > 0]
        
        if len(valid_results) == 0:
            raise ValueError("No valid subjects found!")
        
        all_features, all_labels = zip(*valid_results)
        concat_features = np.concatenate(all_features, axis=0)
        concat_labels = np.concatenate(all_labels, axis=0)

        return concat_features, concat_labels

    def _process_single_subject(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single subject's data to extract features and labels.
        
        Args:
            subject_id: Unique identifier for the subject.
            
        Returns:
            Tuple containing feature windows and corresponding labels for the subject.
        """
        labels_timestamps, labels = self._load_subject_labels(subject_id)
        hr_timestamps, hr_data = self._load_subject_heart_rate(subject_id)
        accelerometer_timestamps, accelerometer_data = self._load_subject_motion(subject_id)

        # Filter out unknown labels
        valid_label_mask = labels > self.unknown_label
        labels_timestamps = labels_timestamps[valid_label_mask]
        labels = labels[valid_label_mask]

        # Merge N3 and N4 into NREM, remap REM
        labels[(labels > 0) & (labels < 5)] = self.label_merge_value
        labels[labels == 5] = self.label_rem_value

        # Check for timestamps reset in heart rate data
        hr_diffs = np.diff(hr_timestamps)
        hr_reset_indices = np.where(hr_diffs < 0)[0]

        if len(hr_reset_indices) > 0:
            end_position = hr_reset_indices[0]
            hr_data = hr_data[:end_position + 1]
            hr_timestamps = hr_timestamps[:end_position + 1]

        # Select only data within the PSG recording period
        psg_start = labels_timestamps[0] - self.lookback_window
        psg_end = labels_timestamps[-1]

        # Create masks to filter the data
        hr_mask = (hr_timestamps >= psg_start) & (hr_timestamps <= psg_end)
        filtered_hr_data = hr_data[hr_mask]

        motion_mask = (accelerometer_timestamps >= psg_start) & (accelerometer_timestamps <= psg_end)
        filtered_motion_data = accelerometer_data[motion_mask]

        # Compute motion magnitude
        motion_magnitude = ((filtered_motion_data ** 2).sum(-1) ** 0.5 - self.motion_gravity_constant).clip(0)

        # Resample data to 1 Hz
        target_resample_timestamp = int(labels_timestamps[-1] - labels_timestamps[0]) + self.psg_buffer + self.lookback_window

        motion_data_resampled = resample_data(motion_magnitude, target_resample_timestamp)
        hr_data_resampled = resample_data(filtered_hr_data, target_resample_timestamp)

        # Normalize heart rate data
        subject_hr_mean = hr_data_resampled.mean()
        subject_hr_std = hr_data_resampled.std()
        hr_normalized = (hr_data_resampled - subject_hr_mean) / (subject_hr_std + self.normalization_epsilon)

        # Compute rolling standard deviations for different window sizes
        hr_data_std_devs = np.stack(
            [compute_rolling_std(hr_normalized, window_size, self.min_samples_for_std) for window_size in self.hr_window_sizes],
            axis=-1
        )

        # Stack motion and heart rate features
        features = np.concatenate([motion_data_resampled.reshape(-1, 1), hr_data_std_devs], axis=-1)

        # Create segments based on lookback window
        segment_end_indices = (labels_timestamps - psg_start).astype(np.int64)
        segment_start_indices = (segment_end_indices - self.lookback_window).astype(np.int64)

        # Extract feature windows
        features_windows = [features[start:end, :] for start, end in zip(segment_start_indices, segment_end_indices)]

        # Filter out incomplete segments
        valid_segments_mask = [f.shape[0] == self.lookback_window for f in features_windows]
        features_windows = [f for f, valid in zip(features_windows, valid_segments_mask) if valid]
        
        if len(features_windows) == 0:
            return np.array([]), np.array([])
        
        features_windows = np.stack(features_windows)
        labels = labels[valid_segments_mask]

        return features_windows, labels

    def _load_subject_labels(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load sleep stage labels for a single subject.
        
        Args:
            subject_id: Unique identifier for the subject.
            
        Returns:
            Tuple containing timestamps and corresponding sleep stage labels.
        """
        file_path = os.path.join(self.main_data_path, 'labels', f'{subject_id}_labeled_sleep.txt')
        data = read_txt_file(file_path)
        timestamps, labels = zip(*data)

        return np.array(timestamps), np.array(labels)

    def _load_subject_heart_rate(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load heart rate data for a single subject.
        
        Args:
            subject_id: Unique identifier for the subject.
            
        Returns:
            Tuple containing timestamps and heart rate measurements (BPM).
        """
        file_path = os.path.join(self.main_data_path, 'heart_rate', f'{subject_id}_heartrate.txt')
        data = read_txt_file(file_path)
        timestamps, hr_bpm = zip(*data)
    
        return np.array(timestamps), np.array(hr_bpm)

    def _load_subject_motion(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load accelerometer motion data for a single subject.
        
        Args:
            subject_id: Unique identifier for the subject.
            
        Returns:
            Tuple containing timestamps and 3D accelerometer measurements.
        """
        file_path = os.path.join(self.main_data_path, 'motion', f'{subject_id}_acceleration.txt')
        data = read_txt_file(file_path)
        timestamps, motion = zip(*data)

        return np.array(timestamps), np.array(motion)

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.
        
        Returns:
            Number of samples.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve.
            
        Returns:
            Tuple containing feature tensor and label tensor.
        """
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )
