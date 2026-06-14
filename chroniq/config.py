import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Make directories if they don't exist
for d in [OUTPUT_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Data preprocessing parameters
DEFAULT_DATASET = "AEP_hourly.csv"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Feature engineering options
LAG_HOURS = [1, 2, 24, 168]  # Lags in hours (1h, 2h, 1 day, 1 week)
ROLLING_WINDOWS = [24, 168]  # Rolling window sizes in hours (1 day, 1 week)

# LSTM Hyperparameters
LSTM_SEQ_LEN = 24           # Use past 24 hours of data to predict the next hour
LSTM_BATCH_SIZE = 64
LSTM_EPOCHS = 15
LSTM_LEARNING_RATE = 0.001
LSTM_PATIENCE = 3            # Early stopping patience
LSTM_UNITS = [64, 32]       # Hidden units in LSTM layers

# ARIMA Hyperparameters
ARIMA_ORDER = (2, 1, 1)      # (p, d, q) order
ARIMA_TRAIN_LIMIT = 2000     # Limit ARIMA fitting to the last N hours of train set for speed
