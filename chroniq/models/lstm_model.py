import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from chroniq.models.base import BaseForecaster
from chroniq.config import LSTM_SEQ_LEN, LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_LEARNING_RATE, LSTM_PATIENCE, LSTM_UNITS
from chroniq.data.preprocessing import prepare_lstm_windows

class LSTMForecaster(BaseForecaster):
    """
    LSTM recurrent neural network forecaster using TensorFlow/Keras.
    """
    def __init__(self, seq_len=LSTM_SEQ_LEN, feature_cols=None, target_col='demand',
                 batch_size=LSTM_BATCH_SIZE, epochs=LSTM_EPOCHS, lr=LSTM_LEARNING_RATE,
                 patience=LSTM_PATIENCE, units=LSTM_UNITS):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.units = units
        self.model = None
        self.history = None
        
    def _build_model(self, num_features):
        model = Sequential()
        # First LSTM layer
        if len(self.units) > 1:
            model.add(LSTM(units=self.units[0], return_sequences=True, input_shape=(self.seq_len, num_features)))
            model.add(Dropout(0.15))
            # Second LSTM layer
            model.add(LSTM(units=self.units[1], return_sequences=False))
        else:
            model.add(LSTM(units=self.units[0], return_sequences=False, input_shape=(self.seq_len, num_features)))
            
        model.add(Dropout(0.15))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=1))  # Predict single continuous value
        
        optimizer = Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def fit(self, train_data, val_data=None):
        """
        Fits the LSTM model on sequential windows.
        train_data: scaled pd.DataFrame
        val_data: scaled pd.DataFrame (optional)
        """
        if self.feature_cols is None:
            self.feature_cols = [c for c in train_data.columns if c not in ['Datetime', self.target_col]]
            
        num_features = len(self.feature_cols)
        print(f"Building LSTM model with {num_features} features...")
        self.model = self._build_model(num_features)
        self.model.summary()
        
        # Prepare training windows
        print("Preparing LSTM training windows...")
        X_train, y_train = prepare_lstm_windows(train_data, self.seq_len, self.feature_cols, self.target_col)
        
        callbacks = []
        val_tuple = None
        
        if val_data is not None:
            print("Preparing LSTM validation windows...")
            # To predict all of val_data, prepend the last seq_len entries of train_data
            combined_val = pd.concat([train_data.iloc[-self.seq_len:], val_data]).reset_index(drop=True)
            X_val, y_val = prepare_lstm_windows(combined_val, self.seq_len, self.feature_cols, self.target_col)
            val_tuple = (X_val, y_val)
            
            # Setup Early Stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=self.payout_patience_or_similar(),
                restore_best_weights=True
            )
            callbacks.append(early_stop)
            
        print(f"Training LSTM model (Max Epochs: {self.epochs}, Batch Size: {self.batch_size})...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=val_tuple,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        print("LSTM model training completed.")
        
    def payout_patience_or_similar(self):
        return self.patience

    def predict(self, test_data, history_data=None):
        """
        Generates predictions for test_data.
        If history_data is provided (e.g. the end of train/val data), it is prepended
        to test_data to ensure predictions are made for every single timestep in test_data.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before calling predict.")
            
        if history_data is not None:
            # Prepend history_data (last seq_len rows) to align predictions exactly with test_data
            combined = pd.concat([history_data.iloc[-self.seq_len:], test_data]).reset_index(drop=True)
            X_test, _ = prepare_lstm_windows(combined, self.seq_len, self.feature_cols, self.target_col)
        else:
            # Predict only on test windows (will have len(test_data) - seq_len predictions)
            X_test, _ = prepare_lstm_windows(test_data, self.seq_len, self.feature_cols, self.target_col)
            
        print(f"Generating LSTM predictions for {len(X_test)} windows...")
        predictions = self.model.predict(X_test, batch_size=self.batch_size)
        return predictions.flatten()
