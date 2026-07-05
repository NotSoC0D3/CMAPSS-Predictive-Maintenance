import os
import pandas as pd

def load_cmapss_data():
    """Loads the CMAPSS FD001 dataset using dynamic relative paths."""
    

    column_names = ['unit_number', 'time_in_cycles'] + \
                   [f'op_setting_{i}' for i in range(1, 4)] + \
                   [f'sensor_{i}' for i in range(1, 22)]
                   

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 3. Build the exact paths to your raw data files
    train_path = os.path.join(BASE_DIR, 'data', 'raw', 'train_FD001.txt')
    test_path = os.path.join(BASE_DIR, 'data', 'raw', 'test_FD001.txt')
    rul_path = os.path.join(BASE_DIR, 'data', 'raw', 'RUL_FD001.txt')

    
    df1 = pd.read_csv(train_path, sep=r'\s+', header=None, names=column_names).dropna(axis=1)
    t1 = pd.read_csv(test_path, sep=r'\s+', header=None, names=column_names).dropna(axis=1)
    v1 = pd.read_csv(rul_path, sep=r'\s+', header=None).dropna(axis=1)
    
    return df1, t1, v1

if __name__ == "__main__":

    train_df, test_df, rul_df = load_cmapss_data()
    print(f"Train Data Shape: {train_df.shape}")
    print(f"Test Data Shape: {test_df.shape}")
    print(f"RUL Data Shape: {rul_df.shape}")