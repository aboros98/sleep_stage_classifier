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

Configuration in `src/configs/config.py`:
- **Model**: `model_type` ("transformer"), `batch_size` (64), `lookback_window` (512 sec)
- **Training**: `max_epochs` (50), `learning_rate` (0.001), `loss_type` ("focal")

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