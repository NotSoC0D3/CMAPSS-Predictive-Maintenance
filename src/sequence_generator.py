import numpy as np

def create_sequences(df, sensor_cols, seq_len=30, target_col='RUL'):
    """
    Creates overlapping sliding window sequences per engine for LSTM input.
    
    Returns:
        X: np.array of shape (num_sequences, seq_len, num_features)
        y: np.array of shape (num_sequences,)
    """
    print(f"Creating sequences of length {seq_len}...")
    X_list, y_list = [], []

    for unit_id, group in df.groupby('unit_number'):
        group = group.sort_values('time_in_cycles') 
        sensor_values = group[sensor_cols].values
        targets = group[target_col].values

        if len(group) < seq_len:
            # Skip units with too few cycles for the given sequence length
            continue

        # Create sliding windows over the unit cycles
        for start_idx in range(len(group) - seq_len + 1):
            end_idx = start_idx + seq_len
            seq_x = sensor_values[start_idx:end_idx]
            seq_y = targets[end_idx - 1] 

            X_list.append(seq_x)
            y_list.append(seq_y)

    return np.array(X_list), np.array(y_list)