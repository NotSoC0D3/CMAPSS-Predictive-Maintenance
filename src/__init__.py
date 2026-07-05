# src/__init__.py

# 1. Data Ingestion
from .data_load import load_cmapss_data

# 2. Data Engineering & Signal Processing
from .preprocess import (
    add_rul_targets, 
    perform_feature_selection, 
    fit_and_scale_data,
    apply_kalman_smoothing,
    create_validation_split
)

# 3. Sequence Formatting
from .sequence_generator import create_sequences

# 4. Model Architectures
from .models import (
    build_baseline_lstm, 
    build_bilstm, 
    build_lstm_attention,
    build_bilstm_attention
)

# 4. Define the Public API
__all__ = [
    "load_cmapss_data",
    "add_rul_targets",
    "perform_feature_selection",
    "fit_and_scale_data",
    "apply_kalman_smoothing",
    "create_validation_split",
    "apply_dwt_feature_engineering",
    "create_sequences",
    "build_baseline_lstm",
    "build_bilstm",
    "build_lstm_attention",
    "build_bilstm_attention"
]