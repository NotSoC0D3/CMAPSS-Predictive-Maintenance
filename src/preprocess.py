import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit
from pykalman import KalmanFilter
import pywt
import pickle
import os

def add_rul_targets(train_df, test_df, rul_df, max_rul=125):
    print("Calculating RUL targets...")
    rul_per_engine = train_df.groupby('unit_number')['time_in_cycles'].max()
    train_df['max_cycle'] = train_df['unit_number'].map(rul_per_engine)
    train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
    train_df['RUL'] = train_df['RUL'].clip(upper=max_rul)
    train_df = train_df.drop(columns=['max_cycle'])

    rul_df.columns = ['RUL']
    last_cycles = test_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    last_cycles['RUL'] = rul_df['RUL']
    rul_dict = dict(zip(last_cycles['unit_number'], last_cycles['RUL']))
    
    test_df['last_cycle_rul'] = test_df['unit_number'].map(rul_dict)
    test_df['max_cycle_so_far'] = test_df.groupby('unit_number')['time_in_cycles'].transform('max')
    test_df['RUL'] = test_df['max_cycle_so_far'] + test_df['last_cycle_rul'] - test_df['time_in_cycles']
    test_df['RUL'] = test_df['RUL'].clip(upper=max_rul)
    test_df = test_df.drop(columns=['last_cycle_rul', 'max_cycle_so_far'])
    
    print("RUL targets added successfully.")
    return train_df, test_df

def perform_feature_selection(train_df, test_df, var_thresh=0.01, corr_thresh=0.2):
    print("Performing feature selection...")
    drop_cols = [
        'sensor_1', 'sensor_5', 'sensor_6', 'sensor_8', 'sensor_10',
        'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16', 'sensor_18', 'sensor_19'
    ]
    train_df = train_df.drop(columns=drop_cols, errors='ignore')
    test_df = test_df.drop(columns=drop_cols, errors='ignore')
    
    sensor_cols = [col for col in train_df.columns if 'sensor' in col]
    variances = train_df[sensor_cols].var()
    correlations = train_df[sensor_cols].corrwith(train_df['RUL']).abs()
    
    priority_sensors = variances[variances > var_thresh].index.intersection(
        correlations[correlations > corr_thresh].index
    ).tolist()
    
    print(f"Dropped {len(drop_cols)} redundant/flat sensors.")
    print(f"Identified priority sensors: {priority_sensors}")
    return train_df, test_df, priority_sensors

def apply_kalman_smoothing(train_df, test_df, priority_sensors):
    print("Applying Kalman Filters to priority sensors...")
    def filter_signal(signal):
        kf = KalmanFilter(
            transition_matrices=[1], observation_matrices=[1],
            initial_state_mean=signal.iloc[0], initial_state_covariance=1,
            observation_covariance=1, transition_covariance=0.01
        )
        state_means, _ = kf.filter(signal.values)
        return state_means.flatten()

    for col in priority_sensors:
        for unit_id in train_df['unit_number'].unique():
            idx = train_df['unit_number'] == unit_id
            train_df.loc[idx, col] = filter_signal(train_df.loc[idx, col])
        for unit_id in test_df['unit_number'].unique():
            idx = test_df['unit_number'] == unit_id
            test_df.loc[idx, col] = filter_signal(test_df.loc[idx, col])
            
    return train_df, test_df

def create_validation_split(train_df, test_df, test_size=0.2):
    print("Creating validation split (GroupShuffleSplit)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, val_idx = next(gss.split(train_df, groups=train_df['unit_number']))
    
    df_train = train_df.iloc[train_idx].reset_index(drop=True)
    df_val = train_df.iloc[val_idx].reset_index(drop=True)
    
    return df_train, df_val, test_df

def apply_dwt_feature_engineering(train_df, val_df, test_df, sensor_cols, wavelet='db4', max_level=1):
    print("Extracting and applying DWT features...")
    def extract_dwt(df):
        dwt_features = []
        for unit_id, group in df.groupby('unit_number'):
            group = group.sort_values('time_in_cycles')
            dwt_dict = {'unit_number': unit_id}
            for col in sensor_cols:
                series = group[col].values
                length = len(series)
                level = min(max_level, pywt.dwt_max_level(length, pywt.Wavelet(wavelet).dec_len))
                if level >= 1:
                    coeffs = pywt.wavedec(series, wavelet, level=level)
                    for l, coeff in enumerate(coeffs):
                        dwt_dict[f'{col}_dwt_L{l}_mean'] = np.mean(coeff)
                        dwt_dict[f'{col}_dwt_L{l}_std'] = np.std(coeff)
            dwt_features.append(dwt_dict)
        return pd.DataFrame(dwt_features).fillna(0)

    train_df = train_df.merge(extract_dwt(train_df), on='unit_number', how='left')
    val_df = val_df.merge(extract_dwt(val_df), on='unit_number', how='left')
    test_df = test_df.merge(extract_dwt(test_df), on='unit_number', how='left')
    
    cols_to_drop = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    train_df = train_df.drop(columns=cols_to_drop, errors='ignore')
    val_df = val_df.drop(columns=cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=cols_to_drop, errors='ignore')

    print("DWT features successfully merged.")
    return train_df, val_df, test_df

def fit_and_scale_data(train_df, val_df, test_df, sensor_cols):
    print("Scaling sensor data...")
    scaler = MinMaxScaler()
    scaler.fit(train_df[sensor_cols])
    
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()
    
    train_scaled[sensor_cols] = scaler.transform(train_df[sensor_cols])
    val_scaled[sensor_cols] = scaler.transform(val_df[sensor_cols])
    test_scaled[sensor_cols] = scaler.transform(test_df[sensor_cols])
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scaler_path = os.path.join(base_dir, 'weights', 'min_max_scaler.pkl')
    
    # Ensure weights directory exists before saving
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Scaler saved successfully to {scaler_path}")
    return train_scaled, val_scaled, test_scaled, scaler