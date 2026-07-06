# Predictive Maintenance on C-MAPSS Using BiLSTM & Multi-Head Attention

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

##  Overview
This repository contains a deep learning framework for predicting the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS FD001 dataset. The project evaluates the efficacy of advanced signal processing combined with recurrent neural networks and self-attention mechanisms.

The final pipeline utilizes 1D Kalman Filtering and Level-1 Discrete Wavelet Transforms (DWT) to extract high-fidelity frequency features, feeding into a custom **Bidirectional LSTM + Multi-Head Attention** architecture to achieve highly accurate RUL forecasting.

##  Repository Structure

```text
├── data/
│   └── raw/                    # Contains C-MAPSS subsets (train_FD001.txt, test_FD001.txt)
├── notebooks/
│   └── CMAPSS.ipynb # EDA, feature selection, signal processing, training, evaluation and visualization
├── src/
│   ├── __init__.py             # Library router
│   ├── data_load.py            # Data ingestion
|   ├── models.py               # Neural network architectures (LSTM, BiLSTM, Attention)
│   ├── preprocess.py           # Kalman filters, DWT, and scaling
│   ├── sequence_generator.py   # Sliding window formatting
│   └── train.py                # Main execution and training orchestrator
├── weights/                    # Saved .keras models and MinMaxScaler (.pkl)
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

#  Best Model Architecture

The primary model executed in this repository is a custom BiLSTM + Attention network.

1. Temporal Context: Two stacked Bidirectional LSTM layers capture degradation patterns from both past and future operational states.

2. Dynamic Focus: A 4-head Multi-Head Self-Attention mechanism dynamically weighs the most critical operational cycles immediately just before failure.

3. Stability: Residual connections bypass the attention block to prevent vanishing gradients problem during early training epochs.

#  Key Results

Model performance is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). The integration of self-attention significantly improved the network's ability to isolate failure patterns.

| Model | MAE (cycles) | RMSE (cycles) |
| :--- | :--- | :--- |
| Baseline Stacked LSTM | 9.17 | 14.34 |
| **BiLSTM** | 8.91 | **12.73** |
| LSTM + Attention | 8.96 | 13.19 |
| **BiLSTM + Attention** | **8.48** | 13.44 |
