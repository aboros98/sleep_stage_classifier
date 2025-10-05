import numpy as np


def compute_rolling_std(data: np.ndarray, window_size: int, min_samples: int = 2) -> np.ndarray:
    """
    Compute rolling standard deviation using only past data (causal).
    
    For each time point, computes standard deviation over the preceding window_size samples,
    excluding the current point. NaN values at the start are forward-filled.
    
    Args:
        data: 1D array of time series data.
        window_size: Number of past samples to include in each window.
        min_samples: Minimum number of samples required to compute std.
        
    Returns:
        Array of rolling standard deviations with the same length as input data.
    """
    result = np.full(len(data), np.nan)
    
    for i in range(len(data)):
        start_idx = max(0, i - window_size)
        window = data[start_idx:i]
        
        if len(window) >= min_samples:
            result[i] = np.std(window)
    
    # Forward fill NaN values at the start
    first_valid = np.where(~np.isnan(result))[0]
    if len(first_valid) > 0:
        result[:first_valid[0]] = result[first_valid[0]]
    
    return result