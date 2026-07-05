import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from src import (
    load_cmapss_data,
    add_rul_targets,
    perform_feature_selection,
    apply_kalman_smoothing,
    create_validation_split,
    apply_dwt_feature_engineering,
    fit_and_scale_data,
    create_sequences,
    build_bilstm_attention # -< only trains this champion
)

def main():
    print("--- STARTING CMAPSS PREDICTIVE MAINTENANCE PIPELINE ---\n")

    # 1. Data Ingestion
    print("[1/5] Loading raw data...")
    train_df, test_df, rul_df = load_cmapss_data()

    # 2. Preprocessing & Feature Engineering
    print("\n[2/5] Engineering features and targets...")
    train_df, test_df = add_rul_targets(train_df, test_df, rul_df)
    train_df, test_df, priority_sensors = perform_feature_selection(train_df, test_df)
    train_df, test_df = apply_kalman_smoothing(train_df, test_df, priority_sensors)
    
    # Split validation set before DWT to prevent leakage
    train_df, val_df, test_df = create_validation_split(train_df, test_df)
    
    # Apply Wavelets
    sensor_cols = [col for col in train_df.columns if 'sensor' in col]
    train_df, val_df, test_df = apply_dwt_feature_engineering(train_df, val_df, test_df, sensor_cols)

    # 3. Scaling & Sequence Generation
    print("\n[3/5] Scaling and generating sequences...")
    # Dynamically grab all feature columns (excluding identifiers and target)
    feature_cols = [col for col in train_df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL']]
    
    train_scaled, val_scaled, test_scaled, _ = fit_and_scale_data(train_df, val_df, test_df, feature_cols)
    
    seq_len = 30
    X_train, y_train = create_sequences(train_scaled, feature_cols, seq_len=seq_len)
    X_val, y_val = create_sequences(val_scaled, feature_cols, seq_len=seq_len)
    X_test, y_test = create_sequences(test_scaled, feature_cols, seq_len=seq_len)

    print(f"Final Training Shape: {X_train.shape}")
    print(f"Final Validation Shape: {X_val.shape}")

    # 4. Model Building
    print("\n[4/5] Compiling BiLSTM + Attention Model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_bilstm_attention(input_shape)

    # 5. Training
    print("\n[5/5] Commencing Training Loop...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = os.path.join(base_dir, 'weights', 'best_bilstm_attention.keras')
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ModelCheckpoint(weights_path, save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        epochs=50,
        batch_size=64,
        verbose=1
    )
    
    print(f"\n--- TRAINING COMPLETE ---")
    print(f"Best model weights saved to: {weights_path}")

if __name__ == "__main__":
    main()