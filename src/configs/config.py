from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    """
    Get the default configuration for training.
    
    Returns:
        ConfigDict containing all training hyperparameters and settings.
    """
    config = ConfigDict()

    # ============================================================================
    # BASIC SETTINGS
    # General configuration for paths, seeds, and parallelization
    # ============================================================================
    config.seed = 1337
    config.main_data_path = "./data"
    config.artifacts_path = "./artifacts"
    config.val_split = 0.1
    config.n_jobs = -1  # Number of parallel jobs (-1 = all CPUs)

    # ============================================================================
    # DATA LOADING & PREPROCESSING
    # Configuration for dataset construction and data loading
    # ============================================================================
    config.batch_size = 64
    config.num_workers = 4
    config.shuffle = True
    config.lookback_window = 512  # Seconds of history to use for prediction
    config.hr_window_sizes = [30, 60, 120]  # Window sizes for HR rolling std (seconds)
    config.epoch_duration = 30  # Sleep epoch duration (seconds)
    config.psg_buffer = 30  # Buffer time after PSG end (seconds)
    config.n_features = 4  # Number of features for inference (1 motion + 3 HR std features)

    # ============================================================================
    # SIGNAL PROCESSING THRESHOLDS
    # Thresholds for motion detection and data processing
    # ============================================================================
    config.motion_gravity_constant = 1.0  # Gravity constant for motion magnitude
    config.default_hr_value = 60.0  # Default HR value when data is missing
    config.min_samples_for_std = 2  # Minimum samples required for std calculation
    config.normalization_epsilon = 1e-6  # Epsilon for numerical stability in normalization
    config.class_weight_epsilon = 1e-6  # Epsilon for class weight computation

    # ============================================================================
    # LABEL CONFIGURATION
    # Label values and mappings for sleep stage classification
    # ============================================================================
    config.unknown_label = -1
    config.label_merge_n3_n4_to_nrem = 1
    config.label_rem_remapped = 2

    # ============================================================================
    # LEARNING RATE SCHEDULER
    # Configuration for learning rate scheduling during training
    # ============================================================================
    config.scheduler = True
    config.warmup_percent = 0.15

    # ============================================================================
    # LOSS FUNCTION
    # Loss function type and parameters for training
    # ============================================================================
    config.loss_type = "focal"
    config.focal_gamma = 2.0
    config.use_class_weights = True
    config.ignore_index = -1
    config.label_smoothing = 0.0

    # ============================================================================
    # LSTM MODEL ARCHITECTURE
    # Architecture parameters for LSTM-based sleep stage classifier
    # ============================================================================
    config.num_classes = 3
    config.model_type = "transformer"  # Model type: "lstm" or "transformer"
    
    # LSTM-specific parameters
    config.num_lstm_layers = 2
    config.hidden_dim = 128
    config.conv_out_channels = 64
    config.dropout_p = 0.2
    config.lstm_bidirectional = True
    config.lstm_dropout = 0.0
    
    # Convolutional layers (feature extraction before LSTM)
    config.conv1_kernel_size = 7
    config.conv1_stride = 4
    config.conv1_padding = 3
    config.conv2_kernel_size = 5
    config.conv2_stride = 4
    config.conv2_padding = 2
    config.conv3_kernel_size = 3
    config.conv3_stride = 3
    config.conv3_padding = 1
    config.conv_channel_multiplier = 2
    
    # ============================================================================
    # TRANSFORMER MODEL ARCHITECTURE
    # Architecture parameters for Transformer-based sleep stage classifier
    # ============================================================================
    config.transformer_embed_dim = 32  # Embedding dimension for transformer
    config.transformer_n_heads = 2  # Number of attention heads
    config.transformer_intermediate_dim = 128  # FFN intermediate dimension
    config.transformer_num_layers = 3  # Number of transformer encoder layers
    config.transformer_max_seq_length = config.lookback_window  # Maximum sequence length for positional encoding
    config.transformer_dropout_p = 0.1  # Dropout probability for transformer

    # ============================================================================
    # TRAINING CONFIGURATION
    # General training parameters for PyTorch Lightning
    # ============================================================================
    config.max_epochs = 50
    config.accelerator = "cpu"
    config.device_list = 1
    config.deterministic = True
    config.log_every_n_steps = 10
    config.check_val_every_n_epoch = 1
    config.precision = "bf16-mixed"
    
    # ============================================================================
    # K-FOLD CROSS-VALIDATION
    # Settings for K-fold cross-validation experiments
    # ============================================================================
    config.n_folds = 5
    config.random_seeds = [42, 1337, 999999, 6712, 5213]  # Seeds for K-fold CV

    # ============================================================================
    # OPTIMIZER
    # Optimizer selection and hyperparameters
    # ============================================================================
    config.optimizer_name = "AdamW"
    config.learning_rate = 0.001
    config.weight_decay = 0.0001
    config.betas = (0.9, 0.999)
    
    # ============================================================================
    # CLASS WEIGHTING
    # Parameters for handling class imbalance
    # ============================================================================
    config.class_weight_beta = 0.9999  # Beta for effective number of samples

    # ============================================================================
    # CALLBACKS
    # Configuration for training callbacks (checkpointing, early stopping, logging)
    # ============================================================================
    config.checkpoint_monitor = "val/f1_score"
    config.checkpoint_save_top_k = 1
    config.checkpoint_every_n_epochs = 1
    config.early_stopping_monitor = "val/f1_score"
    config.early_stopping_patience = 15
    config.early_stopping_mode = "max"
    config.confusion_matrix_log_train = True
    config.confusion_matrix_log_val = True
    config.confusion_matrix_log_every_n_epochs = 1
    config.grad_norm_type = 2

    # ============================================================================
    # LOGISTIC REGRESSION BASELINE HYPERPARAMETERS
    # Hyperparameters for Logistic Regression model as a baseline
    # ============================================================================
    config.logreg_max_iter = 3000  # Maximum number of iterations for LogisticRegression
    config.logreg_class_weight = "balanced"  # Class weighting for LogisticRegression
    config.logreg_random_state = 42  # Random seed for LogisticRegression reproducibility

    return config

