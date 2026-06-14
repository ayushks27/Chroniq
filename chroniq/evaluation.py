import numpy as np  # pyrefly: ignore [missing-import]
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score  # pyrefly: ignore [missing-import]

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Handles zero values in true targets by masking or adding epsilon.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Mask zeros to prevent division by zero
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_metrics(y_true, y_pred):
    """
    Calculates key evaluation metrics: MAE, RMSE, MAPE, and R-squared.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def calculate_classification_metrics(y_true, y_pred, threshold):
    """
    Evaluates models on their ability to predict peak demand events (demand > threshold).
    Returns Precision, Recall, and F1-score as percentages.
    """
    y_true_bin = (np.array(y_true) > threshold).astype(int)
    y_pred_bin = (np.array(y_pred) > threshold).astype(int)
    
    # zero_division=0 handles cases where no peak events are predicted
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    return {
        'Precision': precision * 100,
        'Recall': recall * 100,
        'F1': f1 * 100
    }

def print_metrics_summary(model_name, metrics, class_metrics=None):
    """
    Prints the calculated regression and classification metrics.
    """
    print(f"\n=================== {model_name} Evaluation Metrics ===================")
    print(f"  Mean Absolute Error (MAE):       {metrics['MAE']:.3f} MW")
    print(f"  Root Mean Squared Error (RMSE):  {metrics['RMSE']:.3f} MW")
    print(f"  Mean Absolute % Error (MAPE):    {metrics['MAPE']:.3f}%")
    print(f"  R-squared (R2 Score):            {metrics['R2']:.4f}")
    if class_metrics:
        print(f"  ---------------- Peak Grid Events (Top 10% Load) ----------------")
        print(f"  Precision (Avoid False Alarm):   {class_metrics['Precision']:.2f}%")
        print(f"  Recall (Catch Peak Load):        {class_metrics['Recall']:.2f}%")
        print(f"  F1-Score (Peak Detection):       {class_metrics['F1']:.2f}%")
    print("=======================================================================")
