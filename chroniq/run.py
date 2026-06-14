import os
import argparse
import pandas as pd  # pyrefly: ignore [missing-import]
import numpy as np  # pyrefly: ignore [missing-import]
import xgboost as xgb  # pyrefly: ignore [missing-import]
from chroniq.config import (
    DEFAULT_DATASET, DATASET_DIR, OUTPUT_DIR, TRAIN_RATIO, VAL_RATIO, 
    LSTM_SEQ_LEN, ARIMA_ORDER, ARIMA_TRAIN_LIMIT
)
from chroniq.data.preprocessing import (
    load_and_clean_data, engineer_features, split_data, TimeSeriesScaler
)
from chroniq.models.arima_model import ARIMAForecaster
from chroniq.models.lstm_model import LSTMForecaster
from chroniq.evaluation import calculate_metrics, print_metrics_summary, calculate_classification_metrics
from chroniq.report_generator import generate_html_report, save_static_plots

def main():
    parser = argparse.ArgumentParser(description="Chroniq Time Series Electricity Demand Forecasting Pipeline")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=DEFAULT_DATASET,
        help="Name of the dataset CSV file in the dataset directory"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Override LSTM training epochs"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run in quick mode (few epochs, smaller training sizes) for fast validation"
    )
    args = parser.parse_args()

    print("\n" + "="*50)
    print(" CHRONIQ FORECASTING PIPELINE INITIALIZED")
    print("="*50 + "\n")

    # 1. Load and clean raw dataset
    file_path = os.path.join(DATASET_DIR, args.dataset)
    if not os.path.exists(file_path):
        print(f"Error: Dataset file not found at {file_path}")
        available_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
        print(f"Available CSV files in dataset/ directory: {available_files}")
        return

    raw_df, original_target_name = load_and_clean_data(file_path)
    
    # 2. Engineer features
    print("Engineering features (calendar, lags, rolling statistics)...")
    feat_df = engineer_features(raw_df)
    print(f"Engineered dataframe shape: {feat_df.shape}")

    # 3. Split data chronologically
    train_df, val_df, test_df = split_data(feat_df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)
    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Define feature and target columns
    target_col = 'demand'
    feature_cols = [c for c in train_df.columns if c not in ['Datetime', target_col]]
    
    # 4. Scale data (specifically for LSTM and XGBoost features)
    scaler = TimeSeriesScaler()
    scaler.fit(train_df, feature_cols, target_col)
    
    train_df_scaled = scaler.transform(train_df, feature_cols, target_col)
    val_df_scaled = scaler.transform(val_df, feature_cols, target_col)
    test_df_scaled = scaler.transform(test_df, feature_cols, target_col)

    # ==========================================
    # MODEL 1: ARIMA
    # ==========================================
    train_limit = 500 if args.quick else ARIMA_TRAIN_LIMIT
    arima_model = ARIMAForecaster(order=ARIMA_ORDER, train_limit=train_limit)
    arima_model.fit(train_df)
    
    # Calculate peak stress demand threshold (90th percentile of training load)
    peak_threshold = float(np.percentile(train_df[target_col].values, 90))
    print(f"\nPeak stress demand threshold (90th percentile of Train set): {peak_threshold:.2f} MW")

    # Generate predictions on test set (with 95% prediction intervals)
    arima_test_preds, arima_lower, arima_upper = arima_model.predict(test_df)
    
    # Evaluate ARIMA
    arima_metrics = calculate_metrics(test_df[target_col].values, arima_test_preds)
    arima_class = calculate_classification_metrics(test_df[target_col].values, arima_test_preds, peak_threshold)
    arima_metrics.update(arima_class)
    print_metrics_summary("ARIMA (p,d,q)", arima_metrics, arima_class)

    # ==========================================
    # MODEL 2: LSTM (Deep Learning)
    # ==========================================
    epochs = 2 if args.quick else (args.epochs or 15)
    lstm_model = LSTMForecaster(
        seq_len=LSTM_SEQ_LEN,
        feature_cols=feature_cols,
        target_col=target_col,
        epochs=epochs,
        patience=2 if args.quick else 3
    )
    
    # Fit LSTM on scaled training/validation sets
    lstm_model.fit(train_df_scaled, val_df_scaled)
    
    # Predict on scaled test set (prepending scaled val_df to align lengths)
    lstm_scaled_preds = lstm_model.predict(test_df_scaled, history_data=val_df_scaled)
    
    # Invert predictions to raw MW values
    lstm_test_preds = scaler.inverse_transform_target(lstm_scaled_preds)
    
    # Calculate LSTM out-of-sample prediction intervals using validation set residuals
    lstm_val_scaled_preds = lstm_model.predict(val_df_scaled, history_data=train_df_scaled)
    lstm_val_preds = scaler.inverse_transform_target(lstm_val_scaled_preds)
    lstm_val_residuals = val_df[target_col].values - lstm_val_preds
    lstm_std_err = np.std(lstm_val_residuals)
    
    lstm_lower = lstm_test_preds - 1.96 * lstm_std_err
    lstm_upper = lstm_test_preds + 1.96 * lstm_std_err
    
    # Evaluate LSTM
    lstm_metrics = calculate_metrics(test_df[target_col].values, lstm_test_preds)
    lstm_class = calculate_classification_metrics(test_df[target_col].values, lstm_test_preds, peak_threshold)
    lstm_metrics.update(lstm_class)
    print_metrics_summary("LSTM Neural Network", lstm_metrics, lstm_class)

    # ==========================================
    # MODEL 3: XGBoost (Tabular Machine Learning Baseline)
    # ==========================================
    print("Training XGBoost Regressor baseline...")
    xgb_reg = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_reg.fit(train_df[feature_cols], train_df[target_col])
    
    xgb_test_preds = xgb_reg.predict(test_df[feature_cols])
    
    # Train XGBoost quantile estimators for 95% prediction intervals
    print("Training XGBoost quantile estimators for prediction intervals...")
    try:
        xgb_lower_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, objective='reg:quantileerror', quantile_alpha=0.025)
        xgb_lower_model.fit(train_df[feature_cols], train_df[target_col])
        xgb_lower = xgb_lower_model.predict(test_df[feature_cols])

        xgb_upper_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, objective='reg:quantileerror', quantile_alpha=0.975)
        xgb_upper_model.fit(train_df[feature_cols], train_df[target_col])
        xgb_upper = xgb_upper_model.predict(test_df[feature_cols])
    except Exception as e:
        print(f"Warning: XGBoost quantile objective failed ({e}). Using validation residuals fallback.")
        xgb_val_preds = xgb_reg.predict(val_df[feature_cols])
        xgb_val_residuals = val_df[target_col].values - xgb_val_preds
        xgb_std_err = np.std(xgb_val_residuals)
        xgb_lower = xgb_test_preds - 1.96 * xgb_std_err
        xgb_upper = xgb_test_preds + 1.96 * xgb_std_err
    
    # Evaluate XGBoost
    xgb_metrics = calculate_metrics(test_df[target_col].values, xgb_test_preds)
    xgb_class = calculate_classification_metrics(test_df[target_col].values, xgb_test_preds, peak_threshold)
    xgb_metrics.update(xgb_class)
    print_metrics_summary("XGBoost Regressor (Baseline)", xgb_metrics, xgb_class)

    # ==========================================
    # OUTPUTS & VISUALIZATION
    # ==========================================
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'Datetime': test_df['Datetime'],
        'Actual': test_df[target_col],
        'ARIMA_Prediction': arima_test_preds,
        'LSTM_Prediction': lstm_test_preds,
        'XGBoost_Prediction': xgb_test_preds
    })
    output_path = os.path.join(OUTPUT_DIR, f"{args.dataset.replace('.csv', '')}_predictions.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved predictions to CSV: {output_path}")

    print("\nGenerating interactive HTML report...")
    # Get LSTM training history if available
    lstm_history = lstm_model.history.history if lstm_model.history else None
    
    report_filename = f"forecast_{args.dataset.replace('.csv', '')}.html"
    generate_html_report(
        dataset_name=args.dataset,
        actual_dates=test_df['Datetime'],
        actual_values=test_df[target_col].values,
        arima_preds=arima_test_preds,
        lstm_preds=lstm_test_preds,
        xgb_preds=xgb_test_preds,
        arima_metrics=arima_metrics,
        lstm_metrics=lstm_metrics,
        xgb_metrics=xgb_metrics,
        lstm_loss_history=lstm_history,
        xgb_model=xgb_reg,
        feature_cols=feature_cols,
        peak_threshold=peak_threshold,
        arima_intervals=(arima_lower, arima_upper),
        lstm_intervals=(lstm_lower, lstm_upper),
        xgb_intervals=(xgb_lower, xgb_upper),
        output_filename=report_filename
    )
    
    print("\nGenerating 10 static visualization plots...")
    save_static_plots(
        dataset_name=args.dataset,
        actual_dates=test_df['Datetime'].values,
        actual_values=test_df[target_col].values,
        arima_preds=arima_test_preds,
        lstm_preds=lstm_test_preds,
        xgb_preds=xgb_test_preds,
        lstm_loss_history=lstm_history,
        xgb_model=xgb_reg,
        feature_cols=feature_cols,
        peak_threshold=peak_threshold,
        arima_intervals=(arima_lower, arima_upper),
        lstm_intervals=(lstm_lower, lstm_upper),
        xgb_intervals=(xgb_lower, xgb_upper),
        prefix=f"forecast_{args.dataset.replace('.csv', '')}"
    )
    
if __name__ == "__main__":
    main()
