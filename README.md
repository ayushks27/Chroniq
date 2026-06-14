# Chroniq - Advanced Electricity Demand Forecasting & Grid Diagnostics

Chroniq is a state-of-the-art time-series forecasting pipeline and grid diagnostics platform. It trains, evaluates, and compares statistical (**ARIMA**), deep learning (**LSTM**), and machine learning (**XGBoost**) models to predict hourly electricity demand. 

In addition to traditional regression metrics, Chroniq implements **Uncertainty Quantification (95% Prediction Intervals)** and frames a business-critical **Peak Grid Stress Event Classification** evaluation (evaluating F1-Score, Precision, and Recall for load exceeding the 90th percentile).

It outputs both **10 advanced diagnostic PNG plots** and a premium, **interactive, tabbed HTML dashboard** powered by Plotly.js and glassmorphic styling.

---

## 🚀 Key Features

1.  **Multi-Model Architecture:**
    *   **ARIMA (p, d, q):** Statistical baseline model leveraging statsmodels.
    *   **LSTM Recurrent Neural Network:** Deep learning sequence model built with TensorFlow/Keras.
    *   **XGBoost Regressor:** Tabular machine learning baseline utilizing engineered lag, calendar, and rolling features.
2.  **Uncertainty Quantification (95% Prediction Intervals):**
    *   Analytic bounds for ARIMA based on forecast variance.
    *   Quantile regression estimators (`quantile_alpha=[0.025, 0.975]`) for XGBoost.
    *   Out-of-sample validation residuals standard error bands for LSTM.
3.  **Peak Grid Stress Detection (Classification Framing):**
    *   Identifies grid stress hours exceeding the 90th percentile load of the training set.
    *   Evaluates models on **F1-Score**, **Precision** (avoiding false alarms), and **Recall** (reliability in grid protection).
4.  **Interactive, Tabbed HTML Dashboard:**
    *   **Tab 1: Overview** - Interactive forecast line chart with shaded 95% prediction interval bands, residuals histograms, LSTM loss curves, and interactive confusion matrices.
    *   **Tab 2: Seasonality** - Grouped boxplots of actual demand by month, day-of-week demand bar profiles, and hourly weekday vs. weekend profiles.
    *   **Tab 3: Diagnostics** - XGBoost feature importance and actual vs. predicted scatter grids with $y=x$ reference lines.
    *   **Tab 4: Decomposition** - 4-panel additive time-series decomposition (Observed, Trend, Seasonal, Residuals).

---

## 📂 Project Structure

```text
chroniq/
│
├── chroniq/
│   ├── __init__.py
│   ├── config.py             # Hyperparameters, directories, and settings
│   ├── evaluation.py         # Regression (RMSE, MAE) and classification (F1) metrics
│   ├── report_generator.py   # HTML/Plotly and Matplotlib/Seaborn report compiler
│   ├── run.py                # Main pipeline orchestrator
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py  # Cleaning, scaling, and feature engineering
│   └── models/
│       ├── __init__.py
│       ├── base.py           # Base model wrapper class
│       ├── arima_model.py    # ARIMA wrapper
│       └── lstm_model.py     # LSTM neural network wrapper
│
├── dataset/                  # Contains raw CSV data files (AEP_hourly.csv, etc.)
├── outputs/                  # Saved forecast predictions (CSV)
├── reports/                  # Generated HTML dashboards
│   └── plots/                # 10 diagnostic PNG plots
│
├── .gitignore
├── requirements.txt          # Python library dependencies
└── README.md                 # Project documentation
```

---

## 🛠️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd chroniq
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Using Conda (Recommended)
    conda create -n chroniq python=3.10
    conda activate chroniq
    
    # Or using standard venv
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 📈 Usage

You can execute the pipeline from your project root directory:

### 1. Run the Full Pipeline (Default Dataset)
Trains ARIMA, LSTM (15 epochs), and XGBoost models on `AEP_hourly.csv`, compiles forecast files, and outputs all reports:
```bash
python -m chroniq.run
```

### 2. Run in Quick Validation Mode
Limits LSTM epochs to 2 and ARIMA training steps to 500. Perfect for checking code changes and visual report generation speeds:
```bash
python -m chroniq.run --quick
```

### 3. Run on a Specific Dataset
Run the forecast pipeline on other CSV hourly demand datasets:
```bash
python -m chroniq.run --dataset DAYTON_hourly.csv
```

### 4. Open the Interactive Dashboard
Open the compiled HTML report in your browser to inspect the interactive tabs, zoom into dates, toggle model overlays, and explore data distributions:
*   On Windows (PowerShell):
    ```powershell
    Start-Process "reports/forecast_AEP_hourly.html"
    ```
*   On macOS/Linux:
    ```bash
    open reports/forecast_AEP_hourly.html
    ```

---

## ✍️ Author

*   **Purnendu Raghav Srivastava**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](file:///d:/projects/chroniq/LICENSE) file for details.
