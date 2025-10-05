# Sleep Stage Classification

Deep learning models for sleep stage classification using accelerometer and heart rate data. Classifies sleep into Wake, NREM, and REM stages.

## Setup

### Prerequisites
- Docker
- Docker Compose

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aboros98/sleep_stage_classifier
   cd sleep_classification
   ```

2. Build and start the container:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

3. Download data:
   ```bash
   docker-compose exec sleep-classifier bash download_data.sh
   ```

Alternatively, use the setup script:
```bash
./setup.sh
docker-compose up -d
docker-compose exec sleep-classifier bash download_data.sh
```

## Configuration

All configuration parameters are in `src/configs/config.py`. Key parameters:

### Basic Settings
- **`seed`** (1337): Random seed for reproducibility
- **`main_data_path`** ("./data"): Directory containing input data files
- **`artifacts_path`** ("./artifacts"): Directory for saving models and outputs
- **`val_split`** (0.1): Fraction of subjects used for validation (10%)
- **`n_jobs`** (-1): Number of parallel jobs for preprocessing (-1 = use all CPUs)

### Data Configuration
- **`batch_size`** (64): Number of samples per training batch
- **`num_workers`** (4): Number of subprocesses for data loading
- **`lookback_window`** (512): Seconds of historical data used for each prediction
- **`hr_window_sizes`** ([30, 60, 120]): Window sizes in seconds for computing heart rate rolling standard deviation features
- **`epoch_duration`** (30): Duration of each sleep epoch in seconds (standard PSG convention)
- **`psg_buffer`** (30): Buffer time in seconds after PSG recording ends

### Model Architecture
- **`model_type`** ("transformer"): Model architecture - "lstm" or "transformer"
- **`num_classes`** (3): Number of output classes (Wake, NREM, REM)

**LSTM-specific:**
- **`num_lstm_layers`** (2): Number of stacked LSTM layers
- **`hidden_dim`** (128): Hidden state dimension for LSTM
- **`lstm_bidirectional`** (True): Use bidirectional LSTM (processes sequence forward and backward)
- **`conv_out_channels`** (64): Output channels for convolutional feature extraction layers

**Transformer-specific:**
- **`transformer_embed_dim`** (32): Embedding dimension for input features
- **`transformer_n_heads`** (2): Number of attention heads in multi-head attention
- **`transformer_intermediate_dim`** (128): Hidden dimension in feedforward network
- **`transformer_num_layers`** (3): Number of transformer encoder layers
- **`transformer_dropout_p`** (0.1): Dropout probability for regularization

### Training Hyperparameters
- **`max_epochs`** (50): Maximum number of training epochs
- **`learning_rate`** (0.001): Initial learning rate for optimizer
- **`optimizer_name`** ("AdamW"): Optimizer algorithm (AdamW with weight decay)
- **`weight_decay`** (0.0001): L2 regularization coefficient
- **`scheduler`** (True): Enable learning rate scheduling with warmup
- **`warmup_percent`** (0.15): Percentage of training steps for learning rate warmup

### Loss Function
- **`loss_type`** ("focal"): Loss function type - "focal" (better for imbalanced classes) or "cross_entropy"
- **`focal_gamma`** (2.0): Focusing parameter for focal loss (higher = more focus on hard examples)
- **`use_class_weights`** (True): Weight classes inversely proportional to frequency
- **`class_weight_beta`** (0.9999): Beta parameter for effective number of samples calculation
- **`label_smoothing`** (0.0): Label smoothing factor (0 = hard labels, >0 = soft labels)

### K-Fold Cross-Validation
- **`n_folds`** (5): Number of folds for cross-validation
- **`random_seeds`** ([42, 1337, ...]): Random seeds for each fold

### Callbacks & Monitoring
- **`checkpoint_monitor`** ("val/f1_score"): Metric to monitor for saving best model
- **`early_stopping_patience`** (15): Number of epochs without improvement before stopping
- **`log_every_n_steps`** (10): Log training metrics every N batches

## Usage

Run training scripts using Docker:

```bash
# Single model training - trains one model with default config
docker-compose exec sleep-classifier python train.py

# K-fold cross-validation - trains 5 models across different subject splits for robust evaluation
docker-compose exec sleep-classifier python train_kfold.py

# Logistic regression baseline - trains a simple baseline model for comparison
docker-compose exec sleep-classifier python logistic_regression.py
```

### Script Details

- **`train.py`**: Trains a single deep learning model (LSTM or Transformer) using train/validation split. Saves best model checkpoint and metrics.

- **`train_kfold.py`**: Performs 5-fold cross-validation by splitting subjects into folds. Trains 5 separate models and reports mean ± std performance metrics across folds.

- **`logistic_regression.py`**: Trains a logistic regression model as a baseline. Uses engineered features (motion + HR statistics) for comparison with deep learning approaches.

## Inference

```python
from src.inference import SleepStagePredictor
import numpy as np

# Load model
predictor = SleepStagePredictor("artifacts/timestamp/best_model.ckpt")

# Prepare data (example)
hr_data = [65, 68, 70, 72, 69, 66]  # Heart rate values (BPM)
accel_data = [[0.1, 0.2, 9.8], [0.2, 0.1, 9.7]]  # [x, y, z] accelerometer
duration_seconds = 3600  # 1 hour

# Predict sleep stages
predictions = predictor.predict(hr_data, accel_data, duration_seconds)
```

**Classes**: 0=Wake, 1=NREM, 2=REM  
**Input**: Heart rate + 3-axis accelerometer data