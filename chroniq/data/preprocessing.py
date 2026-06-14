import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from chroniq.config import LAG_HOURS, ROLLING_WINDOWS

def load_and_clean_data(file_path):
    """
    Loads hourly electricity demand CSV, cleans duplicates, fills missing hours,
    and returns a cleaned DataFrame with 'demand' and 'Datetime' columns.
    """
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Identify datetime and target columns
    datetime_col = None
    target_col = None
    
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            datetime_col = col
        elif 'mw' in col.lower() or 'load' in col.lower() or 'demand' in col.lower():
            target_col = col
            
    if not datetime_col or not target_col:
        # Fallbacks
        datetime_col = df.columns[0]
        target_col = df.columns[1]
        
    print(f"Detected columns - Datetime: {datetime_col}, Target (Demand): {target_col}")
    
    # Standardize column names
    df = df.rename(columns={datetime_col: 'Datetime', target_col: 'demand'})
    
    # Convert to datetime and sort
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime')
    
    # Remove duplicate timestamps by taking their mean
    # (Typically due to Autumn daylight savings clock shifts repeating 2:00 AM)
    df_clean = df.groupby('Datetime', as_index=False).mean()
    
    # Reindex to full hourly grid to identify missing hours
    min_date = df_clean['Datetime'].min()
    max_date = df_clean['Datetime'].max()
    full_range = pd.date_range(start=min_date, end=max_date, freq='h')
    
    df_clean = df_clean.set_index('Datetime').reindex(full_range)
    df_clean.index.name = 'Datetime'
    
    # Interpolate missing values (linear interpolation)
    missing_count = df_clean['demand'].isnull().sum()
    if missing_count > 0:
        print(f"Interpolating {missing_count} missing hourly readings.")
        df_clean['demand'] = df_clean['demand'].interpolate(method='linear')
        
    df_clean = df_clean.reset_index()
    return df_clean, target_col

def engineer_features(df):
    """
    Creates calendar features, lags, and rolling stats from the demand series.
    """
    df_feat = df.copy()
    
    # 1. Calendar Features
    df_feat['hour'] = df_feat['Datetime'].dt.hour
    df_feat['dayofweek'] = df_feat['Datetime'].dt.dayofweek
    df_feat['month'] = df_feat['Datetime'].dt.month
    df_feat['dayofyear'] = df_feat['Datetime'].dt.dayofyear
    df_feat['is_weekend'] = (df_feat['dayofweek'] >= 5).astype(int)
    
    # Season mapping
    # 1: Winter (Dec-Feb), 2: Spring (Mar-May), 3: Summer (Jun-Aug), 4: Fall (Sep-Nov)
    df_feat['season'] = df_feat['month'].map(lambda m: 1 if m in [12, 1, 2] else (2 if m in [3, 4, 5] else (3 if m in [6, 7, 8] else 4)))
    
    # 2. Lag Features
    for lag in LAG_HOURS:
        df_feat[f'demand_lag_{lag}'] = df_feat['demand'].shift(lag)
        
    # 3. Rolling Features
    for win in ROLLING_WINDOWS:
        df_feat[f'demand_roll_mean_{win}'] = df_feat['demand'].shift(1).rolling(window=win).mean()
        df_feat[f'demand_roll_std_{win}'] = df_feat['demand'].shift(1).rolling(window=win).std()
        
    # Drop rows with NaN values resulting from shift/rolling operations
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat

def split_data(df, train_ratio=0.8, val_ratio=0.1):
    """
    Splits data chronologically into train, val, and test sets.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    
    return train_df, val_df, test_df

class TimeSeriesScaler:
    """
    Helper class to scale multiple features and specifically invert scaling on the target variable.
    """
    def __init__(self):
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        
    def fit(self, train_df, feature_cols, target_col='demand'):
        self.feature_scaler.fit(train_df[feature_cols])
        self.target_scaler.fit(train_df[[target_col]])
        
    def transform(self, df, feature_cols, target_col='demand'):
        scaled_features = self.feature_scaler.transform(df[feature_cols])
        scaled_target = self.target_scaler.transform(df[[target_col]])
        
        # Create a copy and replace columns
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaled_features
        df_scaled[target_col] = scaled_target
        return df_scaled
    
    def fit_transform(self, train_df, feature_cols, target_col='demand'):
        self.fit(train_df, feature_cols, target_col)
        return self.transform(train_df, feature_cols, target_col)
        
    def inverse_transform_target(self, scaled_target_array):
        # Flatten if 1D array
        if len(scaled_target_array.shape) == 1:
            scaled_target_array = scaled_target_array.reshape(-1, 1)
        inverted = self.target_scaler.inverse_transform(scaled_target_array)
        return inverted.flatten()

def prepare_lstm_windows(df, seq_len, feature_cols, target_col='demand'):
    """
    Prepares windows for LSTM input.
    X shape: (samples, seq_len, num_features)
    y shape: (samples, 1)
    """
    X_data = df[feature_cols].values
    y_data = df[target_col].values
    
    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(X_data[i : i + seq_len])
        y.append(y_data[i + seq_len])
        
    return np.array(X), np.array(y).reshape(-1, 1)
