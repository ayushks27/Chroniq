import os
import json
import numpy as np  # pyrefly: ignore [missing-import]
import pandas as pd  # pyrefly: ignore [missing-import]
from chroniq.config import REPORTS_DIR

def generate_html_report(dataset_name, actual_dates, actual_values, 
                         arima_preds, lstm_preds, xgb_preds,
                         arima_metrics, lstm_metrics, xgb_metrics,
                         lstm_loss_history=None, 
                         xgb_model=None, feature_cols=None, peak_threshold=None,
                         arima_intervals=None, lstm_intervals=None, xgb_intervals=None,
                         output_filename="forecast_report.html"):
    """
    Generates an interactive HTML report with forecasting results and metrics.
    Compares ARIMA, LSTM, and XGBoost.
    Uses Plotly.js for charts and a sleek, premium dark-mode theme.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose  # pyrefly: ignore [missing-import]
    from sklearn.metrics import confusion_matrix  # pyrefly: ignore [missing-import]

    plot_limit = 1008 # 6 weeks of hourly data (1008 hours)
    
    dates_str = [d.strftime('%Y-%m-%d %H:%M:%S') for d in actual_dates[-plot_limit:]]
    y_true = [float(x) for x in actual_values[-plot_limit:]]
    y_arima = [float(x) for x in arima_preds[-plot_limit:]]
    y_lstm = [float(x) for x in lstm_preds[-plot_limit:]]
    y_xgb = [float(x) for x in xgb_preds[-plot_limit:]]
    
    # Calculate residuals
    res_arima = [float(x) for x in (np.array(y_true) - np.array(y_arima))]
    res_lstm = [float(x) for x in (np.array(y_true) - np.array(y_lstm))]
    res_xgb = [float(x) for x in (np.array(y_true) - np.array(y_xgb))]

    # Extract 95% Prediction Intervals if available
    if arima_intervals is not None:
        arima_lower = [float(x) for x in arima_intervals[0][-plot_limit:]]
        arima_upper = [float(x) for x in arima_intervals[1][-plot_limit:]]
    else:
        arima_lower, arima_upper = [], []
        
    if lstm_intervals is not None:
        lstm_lower = [float(x) for x in lstm_intervals[0][-plot_limit:]]
        lstm_upper = [float(x) for x in lstm_intervals[1][-plot_limit:]]
    else:
        lstm_lower, lstm_upper = [], []
        
    if xgb_intervals is not None:
        xgb_lower = [float(x) for x in xgb_intervals[0][-plot_limit:]]
        xgb_upper = [float(x) for x in xgb_intervals[1][-plot_limit:]]
    else:
        xgb_lower, xgb_upper = [], []
    
    # Prepare loss history JSON if available
    loss_epochs = []
    train_loss = []
    val_loss = []
    if lstm_loss_history and 'loss' in lstm_loss_history:
        train_loss = [float(x) for x in lstm_loss_history['loss']]
        val_loss = [float(x) for x in lstm_loss_history.get('val_loss', [])]
        loss_epochs = list(range(1, len(train_loss) + 1))
        
    # Check who is the winner
    arima_rmse = arima_metrics.get('RMSE', float('inf'))
    lstm_rmse = lstm_metrics.get('RMSE', float('inf'))
    xgb_rmse = xgb_metrics.get('RMSE', float('inf'))
    
    rmses = {'ARIMA Model': arima_rmse, 'LSTM Neural Network': lstm_rmse, 'XGBoost Regressor': xgb_rmse}
    winner = min(rmses, key=rmses.get)
    
    colors = {
        'ARIMA Model': '#fb923c',      # Orange
        'LSTM Neural Network': '#38bdf8',  # Blue
        'XGBoost Regressor': '#c084fc'    # Purple
    }
    winner_color = colors[winner]
    
    # ------------------ PRECOMPUTATIONS FOR INTERACTIVE GRAPH TABS ------------------
    # 1. Month boxplot data (using all test data)
    dates_pd_all = pd.to_datetime(actual_dates)
    month_names_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    months_list = [month_names_dict[m] for m in dates_pd_all.dt.month]
    actuals_all_list = [float(x) for x in actual_values]
    
    # 2. Weekly day-of-week averages
    temp_df = pd.DataFrame({'Datetime': dates_pd_all, 'demand': actual_values})
    temp_df['dayofweek'] = temp_df['Datetime'].dt.dayofweek
    temp_df['hour'] = temp_df['Datetime'].dt.hour
    
    day_names_dict = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    temp_df['day_name'] = temp_df['dayofweek'].map(day_names_dict)
    day_means = temp_df.groupby('dayofweek')['demand'].mean().reset_index()
    day_means['day_name'] = day_means['dayofweek'].map(day_names_dict)
    
    weekly_profile_data = {
        'labels': list(day_means['day_name']),
        'values': [float(x) for x in day_means['demand']]
    }
    
    # 3. Weekday vs Weekend Hourly profile
    temp_df['is_weekend'] = temp_df['dayofweek'].map(lambda d: "Weekend" if d >= 5 else "Weekday")
    hourly_profile = temp_df.groupby(['is_weekend', 'hour'])['demand'].mean().reset_index()
    weekday_profile = hourly_profile[hourly_profile['is_weekend'] == 'Weekday'].sort_values('hour')
    weekend_profile = hourly_profile[hourly_profile['is_weekend'] == 'Weekend'].sort_values('hour')
    
    daily_profile_data = {
        'hours': list(range(24)),
        'weekday': [float(x) for x in weekday_profile['demand']],
        'weekend': [float(x) for x in weekend_profile['demand']]
    }
    
    # 4. XGBoost Feature Importance
    xgb_feat_importance = {'features': [], 'scores': []}
    if xgb_model is not None and feature_cols is not None:
        try:
            importances = xgb_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_n = min(12, len(feature_cols))
            top_importances = importances[indices[:top_n]]
            top_features = [feature_cols[i] for i in indices[:top_n]]
            xgb_feat_importance = {
                'features': list(top_features)[::-1],
                'scores': [float(x) for x in top_importances][::-1]
            }
        except Exception as e:
            print(f"Warning: Failed to extract XGBoost feature importances for HTML: {e}")
            
    # 5. Confusion Matrices
    if peak_threshold is None:
        peak_threshold = float(np.percentile(actual_values, 90))
        
    y_true_bin = (np.array(actual_values) > peak_threshold).astype(int)
    arima_bin = (np.array(arima_preds) > peak_threshold).astype(int)
    lstm_bin = (np.array(lstm_preds) > peak_threshold).astype(int)
    xgb_bin = (np.array(xgb_preds) > peak_threshold).astype(int)
    
    try:
        cm_arima = confusion_matrix(y_true_bin, arima_bin).tolist()
        cm_lstm = confusion_matrix(y_true_bin, lstm_bin).tolist()
        cm_xgb = confusion_matrix(y_true_bin, xgb_bin).tolist()
    except Exception as e:
        print(f"Warning: Failed to calculate confusion matrices: {e}")
        cm_arima = [[0, 0], [0, 0]]
        cm_lstm = [[0, 0], [0, 0]]
        cm_xgb = [[0, 0], [0, 0]]
        
    confusion_matrices = {
        'arima': cm_arima,
        'lstm': cm_lstm,
        'xgb': cm_xgb,
        'threshold': peak_threshold
    }
    
    # 6. Seasonal Decomposition (on last 168 hours)
    decomp_subset = temp_df.iloc[-168:]
    decomp_dates_str = [d.strftime('%Y-%m-%d %H:%M:%S') for d in decomp_subset['Datetime']]
    decomp_data = None
    try:
        decomposition = seasonal_decompose(decomp_subset['demand'], model='additive', period=24)
        decomp_data = {
            'dates': decomp_dates_str,
            'observed': [float(x) for x in decomposition.observed],
            'trend': [float(x) for x in decomposition.trend.ffill().bfill()],
            'seasonal': [float(x) for x in decomposition.seasonal],
            'resid': [float(x) for x in decomposition.resid.fillna(0.0)]
        }
    except Exception as e:
        print(f"Warning: Seasonal decomposition failed in HTML report: {e}")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chroniq - Interactive Grid Diagnostics Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: rgba(20, 28, 47, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --primary-glow: rgba(56, 189, 248, 0.15);
            --accent-blue: #38bdf8;
            --accent-orange: #fb923c;
            --accent-purple: #c084fc;
            --accent-green: #34d399;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: 'Inter', sans-serif;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.05) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(167, 139, 250, 0.05) 0%, transparent 40%);
            background-attachment: fixed;
            min-height: 100vh;
            padding: 2rem;
            line-height: 1.5;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 1.5rem;
        }}

        .brand {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .brand-logo {{
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 1.5rem;
            color: #fff;
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
        }}

        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }}

        .dataset-badge {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--card-border);
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-size: 0.875rem;
            color: var(--accent-blue);
            font-weight: 500;
        }}

        /* Tab Layout */
        .tabs-nav {{
            display: flex;
            gap: 0.75rem;
            margin-bottom: 2rem;
            background: rgba(20, 28, 47, 0.4);
            border: 1px solid var(--card-border);
            padding: 0.5rem;
            border-radius: 12px;
            backdrop-filter: blur(8px);
        }}

        .tab-btn {{
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            color: var(--text-secondary);
            background: transparent;
            border: none;
            transition: all 0.3s ease;
            font-family: 'Outfit', sans-serif;
            font-size: 0.95rem;
        }}

        .tab-btn:hover {{
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.05);
        }}

        .tab-btn.active {{
            color: #fff;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.25);
        }}

        .tab-content {{
            display: none;
            animation: fadeIn 0.4s ease;
        }}

        .tab-content.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }}

        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }}

        .card:hover {{
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.2);
        }}

        .card-full-width {{
            grid-column: span 4;
        }}

        .card-half-width {{
            grid-column: span 2;
        }}

        .card-three-quarter {{
            grid-column: span 3;
        }}

        .card-one-quarter {{
            grid-column: span 1;
        }}

        .card-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .metric-comparison-box {{
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .model-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 0.5rem;
            margin-top: 0.25rem;
        }}

        .model-name {{
            font-weight: 600;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .model-color-indicator {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }}

        .arima-color {{ background-color: var(--accent-orange); }}
        .lstm-color {{ background-color: var(--accent-blue); }}
        .xgb-color {{ background-color: var(--accent-purple); }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
            margin-top: 0.5rem;
        }}

        .metric-item {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 0.5rem;
            text-align: center;
        }}

        .metric-label {{
            font-size: 0.7rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.15rem;
        }}

        .metric-val {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.1rem;
            font-weight: 700;
        }}

        .best-performer-box {{
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.08), rgba(192, 132, 252, 0.08));
            border: 1px solid rgba(56, 189, 248, 0.15);
            border-radius: 12px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            height: 100%;
        }}

        .best-badge {{
            background: {winner_color}33;
            color: {winner_color};
            border: 1px solid {winner_color}88;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            margin-bottom: 0.75rem;
        }}

        .winner-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            color: #fff;
        }}

        .winner-desc {{
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        .chart-container {{
            width: 100%;
            height: 480px;
        }}

        .small-chart-container {{
            width: 100%;
            height: 280px;
        }}

        .grid-three-col {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
        }}

        @media (max-width: 1024px) {{
            .grid-three-col {{
                grid-template-columns: 1fr;
            }}
        }}

        .footer-note {{
            text-align: center;
            margin-top: 3rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>

    <header>
        <div class="brand">
            <div class="brand-logo">C</div>
            <div>
                <h1>CHRONIQ</h1>
                <p style="font-size: 0.8rem; color: var(--text-secondary)">Interactive Grid Diagnostics & Analytics</p>
            </div>
        </div>
        <div class="dataset-badge">
            Dataset: <strong>{dataset_name}</strong>
        </div>
    </header>

    <!-- Tabbed Navigation Bar -->
    <div class="tabs-nav">
        <button class="tab-btn active" onclick="switchTab('overview')">Forecast Overview</button>
        <button class="tab-btn" onclick="switchTab('temporal')">Temporal & Seasonality</button>
        <button class="tab-btn" onclick="switchTab('diagnostics')">Model Diagnostics</button>
        <button class="tab-btn" onclick="switchTab('decomposition')">Seasonal Decomposition</button>
    </div>

    <!-- ==================== TAB 1: FORECAST OVERVIEW ==================== -->
    <div id="overview" class="tab-content active">
        <!-- Top Summary Row -->
        <div class="dashboard-grid">
            <div class="card card-one-quarter">
                <div class="best-performer-box">
                    <div class="best-badge">Top Performer</div>
                    <div class="winner-title">{winner}</div>
                    <p class="winner-desc">Achieved the lowest RMSE of {min(rmses.values()):.2f} MW on the out-of-sample test set.</p>
                </div>
            </div>

            <div class="card card-three-quarter">
                <div class="card-title">Model Performance Comparison (Peak Threshold: Top 10% Load)</div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem;">
                    <!-- ARIMA Metrics -->
                    <div class="metric-comparison-box">
                        <div class="model-header">
                            <div class="model-name">
                                <span class="model-color-indicator arima-color"></span>
                                ARIMA (Statistical)
                            </div>
                        </div>
                        <div class="metric-grid">
                            <div class="metric-item"><div class="metric-label">RMSE</div><div class="metric-val" style="color: var(--accent-orange)">{arima_metrics['RMSE']:.2f}</div></div>
                            <div class="metric-item"><div class="metric-label">MAE</div><div class="metric-val">{arima_metrics['MAE']:.2f}</div></div>
                            <div class="metric-item"><div class="metric-label">MAPE</div><div class="metric-val">{arima_metrics['MAPE']:.2f}%</div></div>
                            <div class="metric-item"><div class="metric-label">R² Score</div><div class="metric-val">{arima_metrics['R2']:.4f}</div></div>
                            <div class="metric-item"><div class="metric-label">F1 Peak</div><div class="metric-val" style="color: var(--accent-green)">{arima_metrics.get('F1', 0.0):.1f}%</div></div>
                            <div class="metric-item"><div class="metric-label">Prec / Rec</div><div class="metric-val" style="font-size: 0.8rem; padding-top: 0.25rem;">{arima_metrics.get('Precision', 0.0):.1f}% / {arima_metrics.get('Recall', 0.0):.1f}%</div></div>
                        </div>
                    </div>

                    <!-- LSTM Metrics -->
                    <div class="metric-comparison-box">
                        <div class="model-header">
                            <div class="model-name">
                                <span class="model-color-indicator lstm-color"></span>
                                LSTM (Deep Learning)
                            </div>
                        </div>
                        <div class="metric-grid">
                            <div class="metric-item"><div class="metric-label">RMSE</div><div class="metric-val" style="color: var(--accent-blue)">{lstm_metrics['RMSE']:.2f}</div></div>
                            <div class="metric-item"><div class="metric-label">MAE</div><div class="metric-val">{lstm_metrics['MAE']:.2f}</div></div>
                            <div class="metric-item"><div class="metric-label">MAPE</div><div class="metric-val">{lstm_metrics['MAPE']:.2f}%</div></div>
                            <div class="metric-item"><div class="metric-label">R² Score</div><div class="metric-val">{lstm_metrics['R2']:.4f}</div></div>
                            <div class="metric-item"><div class="metric-label">F1 Peak</div><div class="metric-val" style="color: var(--accent-green)">{lstm_metrics.get('F1', 0.0):.1f}%</div></div>
                            <div class="metric-item"><div class="metric-label">Prec / Rec</div><div class="metric-val" style="font-size: 0.8rem; padding-top: 0.25rem;">{lstm_metrics.get('Precision', 0.0):.1f}% / {lstm_metrics.get('Recall', 0.0):.1f}%</div></div>
                        </div>
                    </div>

                    <!-- XGBoost Metrics -->
                    <div class="metric-comparison-box">
                        <div class="model-header">
                            <div class="model-name">
                                <span class="model-color-indicator xgb-color"></span>
                                XGBoost (Tabular ML)
                            </div>
                        </div>
                        <div class="metric-grid">
                            <div class="metric-item"><div class="metric-label">RMSE</div><div class="metric-val" style="color: var(--accent-purple)">{xgb_metrics['RMSE']:.2f}</div></div>
                            <div class="metric-item"><div class="metric-label">MAE</div><div class="metric-val">{xgb_metrics['MAE']:.2f}</div></div>
                            <div class="metric-item"><div class="metric-label">MAPE</div><div class="metric-val">{xgb_metrics['MAPE']:.2f}%</div></div>
                            <div class="metric-item"><div class="metric-label">R² Score</div><div class="metric-val">{xgb_metrics['R2']:.4f}</div></div>
                            <div class="metric-item"><div class="metric-label">F1 Peak</div><div class="metric-val" style="color: var(--accent-green)">{xgb_metrics.get('F1', 0.0):.1f}%</div></div>
                            <div class="metric-item"><div class="metric-label">Prec / Rec</div><div class="metric-val" style="font-size: 0.8rem; padding-top: 0.25rem;">{xgb_metrics.get('Precision', 0.0):.1f}% / {xgb_metrics.get('Recall', 0.0):.1f}%</div></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Big Main Forecast Line Plot -->
        <div class="dashboard-grid">
            <div class="card card-full-width">
                <div class="card-title">
                    Interactive Demand Forecast vs Actuals (Zoomed View: Last 6 Weeks)
                    <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-secondary);">Interactive zoom, pan, and double-click legends to isolate lines</span>
                </div>
                <div id="forecastPlot" class="chart-container"></div>
            </div>
        </div>

        <!-- Residuals and loss curve -->
        <div class="dashboard-grid">
            <div class="card card-half-width">
                <div class="card-title">Forecast Errors (Residuals) Distribution</div>
                <div id="residualsPlot" class="small-chart-container"></div>
            </div>

            <div class="card card-half-width">
                <div class="card-title">LSTM Model Loss Convergence</div>
                {f'<div id="lossPlot" class="small-chart-container"></div>' if lstm_loss_history else '<div style="display:flex;align-items:center;justify-content:center;height:280px;color:var(--text-secondary)">No Training History Available (Quick/Incomplete Mode)</div>'}
            </div>
        </div>

        <!-- Confusion Matrices Row -->
        <div class="dashboard-grid">
            <div class="card card-full-width">
                <div class="card-title">Peak Demand Event Detection Confusion Matrices (Threshold: {confusion_matrices['threshold']:.2f} MW)</div>
                <div class="grid-three-col">
                    <div id="cmArima" style="height: 300px;"></div>
                    <div id="cmLstm" style="height: 300px;"></div>
                    <div id="cmXgb" style="height: 300px;"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- ==================== TAB 2: TEMPORAL & SEASONALITY ==================== -->
    <div id="temporal" class="tab-content">
        <div class="dashboard-grid">
            <div class="card card-half-width">
                <div class="card-title">Monthly Demand Distribution (Boxplot over entire test set)</div>
                <div id="boxPlot" style="height: 350px; width: 100%;"></div>
            </div>

            <div class="card card-half-width">
                <div class="card-title">Weekly Load Profile (Average Demand by Day of Week)</div>
                <div id="weeklyPlot" style="height: 350px; width: 100%;"></div>
            </div>
        </div>

        <div class="dashboard-grid">
            <div class="card card-full-width">
                <div class="card-title">Daily Hourly Profile (Weekday vs Weekend average load cycles)</div>
                <div id="dailyPlot" style="height: 400px; width: 100%;"></div>
            </div>
        </div>
    </div>

    <!-- ==================== TAB 3: MODEL DIAGNOSTICS ==================== -->
    <div id="diagnostics" class="tab-content">
        <div class="dashboard-grid">
            <div class="card card-half-width">
                <div class="card-title">XGBoost Top Feature Importances</div>
                {f'<div id="importancePlot" style="height: 380px; width: 100%;"></div>' if xgb_feat_importance['features'] else '<div style="display:flex;align-items:center;justify-content:center;height:380px;color:var(--text-secondary)">XGBoost Model Feature Importances Not Found</div>'}
            </div>

            <div class="card card-half-width">
                <div class="card-title">Actual vs Predicted Scatter Alignment Grid</div>
                <div style="font-size: 0.9rem; color: var(--text-secondary); display:flex; flex-direction:column; justify-content:center; height:100%; padding: 1rem;">
                    <p style="margin-bottom: 1rem; color: #fff; font-weight: 600;">How to read these charts:</p>
                    <p style="margin-bottom: 0.75rem;">• Points closer to the diagonal red line ($y=x$) represent perfect forecasts.</p>
                    <p style="margin-bottom: 0.75rem;">• Points falling <strong>above</strong> the red line indicate the model overpredicted the demand.</p>
                    <p style="margin-bottom: 0.75rem;">• Points falling <strong>below</strong> the red line indicate the model underpredicted the demand.</p>
                    <p>• Wide spreads represent high variance, while offset centroids represent model bias.</p>
                </div>
            </div>
        </div>

        <div class="dashboard-grid">
            <div class="card card-full-width">
                <div class="card-title">Scatter Alignments Panels</div>
                <div class="grid-three-col">
                    <div id="scatterArima" style="height: 320px;"></div>
                    <div id="scatterLstm" style="height: 320px;"></div>
                    <div id="scatterXgb" style="height: 320px;"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- ==================== TAB 4: TIME-SERIES DECOMPOSITION ==================== -->
    <div id="decomposition" class="tab-content">
        <div class="dashboard-grid">
            <div class="card card-full-width">
                <div class="card-title">
                    Time Series Additive Seasonal Decomposition (Last 7 Days)
                    <span style="font-size: 0.8rem; font-weight: normal; color: var(--text-secondary);">Isolating Trend, daily seasonal cycle, and residual noise from observed load</span>
                </div>
                {f'<div id="decompPlot" style="height: 680px; width: 100%;"></div>' if decomp_data else '<div style="display:flex;align-items:center;justify-content:center;height:680px;color:var(--text-secondary)">Decomposition Not Available</div>'}
            </div>
        </div>
    </div>

    <p class="footer-note">Chroniq Forecasting Diagnostics Platform • Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <script>
        // Data injected from Python
        const dates = {json.dumps(dates_str)};
        const actuals = {json.dumps(y_true)};
        const arima = {json.dumps(y_arima)};
        const lstm = {json.dumps(y_lstm)};
        const xgbData = {json.dumps(y_xgb)};
        
        const resArima = {json.dumps(res_arima)};
        const resLstm = {json.dumps(res_lstm)};
        const resXgb = {json.dumps(res_xgb)};

        // Inject 95% Prediction Interval Data
        const arimaLower = {json.dumps(arima_lower)};
        const arimaUpper = {json.dumps(arima_upper)};
        const lstmLower = {json.dumps(lstm_lower)};
        const lstmUpper = {json.dumps(lstm_upper)};
        const xgbLower = {json.dumps(xgb_lower)};
        const xgbUpper = {json.dumps(xgb_upper)};

        // Tab Switching Logic
        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            event.currentTarget.classList.add('active');
            document.getElementById(tabId).classList.add('active');
            
            // Force Plotly plots to resize to fit containers
            window.dispatchEvent(new Event('resize'));
        }}

        // ------------------ PLOT 1: FORECAST ZOOM & PREDICTION INTERVALS ------------------
        const traceActual = {{
            x: dates,
            y: actuals,
            name: 'Actual Demand',
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#f8fafc', width: 2 }},
            opacity: 0.85
        }};

        const traceArima = {{
            x: dates,
            y: arima,
            name: 'ARIMA Forecast',
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#fb923c', width: 1.5, dash: 'dot' }}
        }};

        const traceLstm = {{
            x: dates,
            y: lstm,
            name: 'LSTM Forecast',
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#38bdf8', width: 1.8 }}
        }};

        const traceXgb = {{
            x: dates,
            y: xgbData,
            name: 'XGBoost Forecast',
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#c084fc', width: 1.5, dash: 'dash' }}
        }};

        // ARIMA 95% PI Traces
        const traceArimaUpper = {{
            x: dates,
            y: arimaUpper,
            type: 'scatter',
            mode: 'lines',
            line: {{ width: 0 }},
            showlegend: false,
            hoverinfo: 'none'
        }};
        const traceArimaLower = {{
            x: dates,
            y: arimaLower,
            type: 'scatter',
            mode: 'lines',
            line: {{ width: 0 }},
            fill: 'tonexty',
            fillcolor: 'rgba(251, 146, 60, 0.12)',
            name: 'ARIMA 95% PI',
            hoverinfo: 'none'
        }};

        // LSTM 95% PI Traces
        const traceLstmUpper = {{
            x: dates,
            y: lstmUpper,
            type: 'scatter',
            mode: 'lines',
            line: {{ width: 0 }},
            showlegend: false,
            hoverinfo: 'none'
        }};
        const traceLstmLower = {{
            x: dates,
            y: lstmLower,
            type: 'scatter',
            mode: 'lines',
            line: {{ width: 0 }},
            fill: 'tonexty',
            fillcolor: 'rgba(56, 189, 248, 0.12)',
            name: 'LSTM 95% PI',
            hoverinfo: 'none'
        }};

        // XGBoost 95% PI Traces
        const traceXgbUpper = {{
            x: dates,
            y: xgbUpper,
            type: 'scatter',
            mode: 'lines',
            line: {{ width: 0 }},
            showlegend: false,
            hoverinfo: 'none'
        }};
        const traceXgbLower = {{
            x: dates,
            y: xgbLower,
            type: 'scatter',
            mode: 'lines',
            line: {{ width: 0 }},
            fill: 'tonexty',
            fillcolor: 'rgba(192, 132, 252, 0.12)',
            name: 'XGBoost 95% PI',
            hoverinfo: 'none'
        }};

        const forecastLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter', color: '#94a3b8' }},
            margin: {{ t: 20, r: 20, l: 60, b: 40 }},
            hovermode: 'x unified',
            xaxis: {{
                gridcolor: 'rgba(255,255,255,0.05)',
                linecolor: 'rgba(255,255,255,0.1)',
                title: 'Timestamp'
            }},
            yaxis: {{
                gridcolor: 'rgba(255,255,255,0.05)',
                linecolor: 'rgba(255,255,255,0.1)',
                title: 'Electricity Load (MW)'
            }},
            legend: {{
                orientation: 'h',
                y: 1.1,
                x: 0.5,
                xanchor: 'center',
                font: {{ size: 12, color: '#f8fafc' }}
            }}
        }};

        const forecastTraces = [];
        if (arimaUpper.length > 0 && arimaLower.length > 0) {{
            forecastTraces.push(traceArimaUpper, traceArimaLower);
        }}
        if (lstmUpper.length > 0 && lstmLower.length > 0) {{
            forecastTraces.push(traceLstmUpper, traceLstmLower);
        }}
        if (xgbUpper.length > 0 && xgbLower.length > 0) {{
            forecastTraces.push(traceXgbUpper, traceXgbLower);
        }}
        forecastTraces.push(traceActual, traceArima, traceLstm, traceXgb);

        Plotly.newPlot('forecastPlot', forecastTraces, forecastLayout);

        // ------------------ PLOT 2: RESIDUALS DENSITY ------------------
        const traceResArima = {{
            x: resArima,
            name: 'ARIMA',
            type: 'histogram',
            opacity: 0.45,
            marker: {{ color: '#fb923c' }}
        }};

        const traceResLstm = {{
            x: resLstm,
            name: 'LSTM',
            type: 'histogram',
            opacity: 0.45,
            marker: {{ color: '#38bdf8' }}
        }};

        const traceResXgb = {{
            x: resXgb,
            name: 'XGBoost',
            type: 'histogram',
            opacity: 0.45,
            marker: {{ color: '#c084fc' }}
        }};

        const residualsLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter', color: '#94a3b8' }},
            margin: {{ t: 20, r: 20, l: 50, b: 40 }},
            barmode: 'overlay',
            showlegend: true,
            legend: {{
                orientation: 'h',
                y: 1.2,
                x: 0.5,
                xanchor: 'center',
                font: {{ color: '#f8fafc' }}
            }},
            xaxis: {{
                gridcolor: 'rgba(255,255,255,0.05)',
                linecolor: 'rgba(255,255,255,0.1)',
                title: 'Error (Actual - Forecast) MW'
            }},
            yaxis: {{
                gridcolor: 'rgba(255,255,255,0.05)',
                linecolor: 'rgba(255,255,255,0.1)',
                title: 'Frequency'
            }}
        }};

        Plotly.newPlot('residualsPlot', [traceResArima, traceResLstm, traceResXgb], residualsLayout);

        // ------------------ PLOT 3: LSTM LOSShistory ------------------
        {f'''
        const lossEpochs = {json.dumps(loss_epochs)};
        const tLoss = {json.dumps(train_loss)};
        const vLoss = {json.dumps(val_loss)};

        const traceTrainLoss = {{
            x: lossEpochs,
            y: tLoss,
            name: 'Train Loss',
            type: 'scatter',
            mode: 'lines+markers',
            line: {{ color: '#a78bfa', width: 2 }}
        }};

        const traceValLoss = {{
            x: lossEpochs,
            y: vLoss,
            name: 'Val Loss',
            type: 'scatter',
            mode: 'lines+markers',
            line: {{ color: '#34d399', width: 2 }}
        }};

        const lossLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter', color: '#94a3b8' }},
            margin: {{ t: 20, r: 20, l: 50, b: 40 }},
            xaxis: {{
                gridcolor: 'rgba(255,255,255,0.05)',
                linecolor: 'rgba(255,255,255,0.1)',
                title: 'Epoch'
            }},
            yaxis: {{
                gridcolor: 'rgba(255,255,255,0.05)',
                linecolor: 'rgba(255,255,255,0.1)',
                title: 'Loss Value (MSE)'
            }},
            legend: {{
                orientation: 'h',
                y: 1.2,
                x: 0.5,
                xanchor: 'center',
                font: {{ color: '#f8fafc' }}
            }}
        }};

        Plotly.newPlot('lossPlot', [traceTrainLoss, traceValLoss], lossLayout);
        ''' if lstm_loss_history else ""}

        // ------------------ PLOT 9: CONFUSION MATRICES ------------------
        const confusionMatrices = {json.dumps(confusion_matrices)};
        
        function drawConfusionMatrix(elementId, zData, titleText, colorScale) {{
            const data = [{{
                z: zData,
                x: ['Normal', 'Peak Event'],
                y: ['Normal', 'Peak Event'],
                type: 'heatmap',
                colorscale: colorScale,
                showscale: false,
                xgap: 4,
                ygap: 4
            }}];
            
            const annotations = [];
            for (let i = 0; i < 2; i++) {{
                for (let j = 0; j < 2; j++) {{
                    annotations.push({{
                        x: j,
                        y: i,
                        text: zData[i][j].toString(),
                        font: {{ family: 'Inter', size: 16, color: '#f8fafc', weight: 'bold' }},
                        showarrow: false
                    }});
                }}
            }}
            
            const layout = {{
                title: {{ text: titleText, font: {{ family: 'Outfit', size: 14, color: '#f8fafc' }} }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ family: 'Inter', color: '#94a3b8' }},
                margin: {{ t: 40, r: 20, l: 85, b: 45 }},
                xaxis: {{
                    ticks: '',
                    side: 'bottom',
                    title: 'Predicted Label',
                    linecolor: 'rgba(255,255,255,0.1)'
                }},
                yaxis: {{
                    ticks: '',
                    title: 'Actual Label',
                    linecolor: 'rgba(255,255,255,0.1)'
                }},
                annotations: annotations
            }};
            
            Plotly.newPlot(elementId, data, layout);
        }}

        drawConfusionMatrix('cmArima', confusionMatrices.arima, 'ARIMA Peak Detection', 'Oranges');
        drawConfusionMatrix('cmLstm', confusionMatrices.lstm, 'LSTM Peak Detection', 'Blues');
        drawConfusionMatrix('cmXgb', confusionMatrices.xgb, 'XGBoost Peak Detection', 'Purples');


        // ------------------ PLOT 4: MONTH BOXPLOT ------------------
        const monthNamesArray = {json.dumps(months_list)};
        const actualsAll = {json.dumps(actuals_all_list)};
        const monthsOrder = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

        const traceBox = {{
            x: monthNamesArray,
            y: actualsAll,
            type: 'box',
            name: 'Load Distribution',
            boxpoints: false,
            marker: {{ color: '#38bdf8' }},
            line: {{ width: 1.5 }}
        }};

        const boxLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter', color: '#94a3b8' }},
            margin: {{ t: 20, r: 20, l: 60, b: 40 }},
            xaxis: {{
                categoryorder: 'array',
                categoryarray: monthsOrder,
                gridcolor: 'rgba(255,255,255,0.05)',
                linecolor: 'rgba(255,255,255,0.1)'
            }},
            yaxis: {{
                gridcolor: 'rgba(255,255,255,0.05)',
                linecolor: 'rgba(255,255,255,0.1)',
                title: 'Demand (MW)'
            }}
        }};

        Plotly.newPlot('boxPlot', [traceBox], boxLayout);


        // ------------------ PLOT 5: WEEKLY LOAD PROFILE (DoW Bar Chart) ------------------
        const weeklyData = {json.dumps(weekly_profile_data)};

        const traceWeekly = {{
            x: weeklyData.labels,
            y: weeklyData.values,
            type: 'bar',
            name: 'Average Load',
            marker: {{
                color: '#fb923c',
                opacity: 0.8,
                line: {{ color: '#fb923c', width: 1 }}
            }}
        }};

        const weeklyLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter', color: '#94a3b8' }},
            margin: {{ t: 20, r: 20, l: 60, b: 40 }},
            xaxis: {{
                gridcolor: 'rgba(255,255,255,0.05)',
                linecolor: 'rgba(255,255,255,0.1)'
            }},
            yaxis: {{
                gridcolor: 'rgba(255,255,255,0.05)',
                linecolor: 'rgba(255,255,255,0.1)',
                title: 'Average Load (MW)'
            }}
        }};

        Plotly.newPlot('weeklyPlot', [traceWeekly], weeklyLayout);


        // ------------------ PLOT 6: DAILY PROFILE (Weekday vs Weekend Hourly) ------------------
        const dailyData = {json.dumps(daily_profile_data)};

        const traceWeekday = {{
            x: dailyData.hours,
            y: dailyData.weekday,
            name: 'Weekday Profile',
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#38bdf8', width: 3 }}
        }};

        const traceWeekend = {{
            x: dailyData.hours,
            y: dailyData.weekend,
            name: 'Weekend Profile',
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#c084fc', width: 3 }}
        }};

        const dailyLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ family: 'Inter', color: '#94a3b8' }},
            margin: {{ t: 20, r: 20, l: 60, b: 40 }},
            xaxis: {{
                tickmode: 'linear',
                tick0: 0,
                dtick: 2,
                gridcolor: 'rgba(255,255,255,0.05)',
                linecolor: 'rgba(255,255,255,0.1)',
                title: 'Hour of Day'
            }},
            yaxis: {{
                gridcolor: 'rgba(255,255,255,0.05)',
                linecolor: 'rgba(255,255,255,0.1)',
                title: 'Average Demand (MW)'
            }},
            legend: {{
                font: {{ color: '#f8fafc' }}
            }}
        }};

        Plotly.newPlot('dailyPlot', [traceWeekday, traceWeekend], dailyLayout);


        // ------------------ PLOT 7: FEATURE IMPORTANCE ------------------
        const xgbImportance = {json.dumps(xgb_feat_importance)};
        
        if (xgbImportance.features && xgbImportance.features.length > 0) {{
            const traceImportance = {{
                x: xgbImportance.scores,
                y: xgbImportance.features,
                type: 'bar',
                orientation: 'h',
                marker: {{
                    color: '#c084fc',
                    opacity: 0.8
                }}
            }};

            const importanceLayout = {{
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ family: 'Inter', color: '#94a3b8' }},
                margin: {{ t: 20, r: 20, l: 150, b: 40 }},
                xaxis: {{
                    gridcolor: 'rgba(255,255,255,0.05)',
                    linecolor: 'rgba(255,255,255,0.1)',
                    title: 'Relative Importance Score'
                }},
                yaxis: {{
                    gridcolor: 'rgba(255,255,255,0.05)',
                    linecolor: 'rgba(255,255,255,0.1)'
                }}
            }};

            Plotly.newPlot('importancePlot', [traceImportance], importanceLayout);
        }}


        // ------------------ PLOT 8: SCATTER ALIGNMENTS ------------------
        const minVal = Math.min(...actuals, ...arima, ...lstm, ...xgbData) * 0.95;
        const maxVal = Math.max(...actuals, ...arima, ...lstm, ...xgbData) * 1.05;

        function drawScatter(elementId, predData, modelColor, modelName) {{
            const tracePoints = {{
                x: actuals,
                y: predData,
                mode: 'markers',
                type: 'scatter',
                name: modelName,
                marker: {{ color: modelColor, opacity: 0.35, size: 5 }}
            }};
            
            const traceRefLine = {{
                x: [minVal, maxVal],
                y: [minVal, maxVal],
                mode: 'lines',
                name: 'y = x Reference',
                line: {{ color: '#ef4444', width: 1.5, dash: 'dash' }}
            }};
            
            const layout = {{
                title: {{ text: modelName + ' Alignment', font: {{ family: 'Outfit', size: 14, color: '#f8fafc' }} }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ family: 'Inter', color: '#94a3b8' }},
                margin: {{ t: 40, r: 20, l: 60, b: 40 }},
                hovermode: 'closest',
                xaxis: {{
                    gridcolor: 'rgba(255,255,255,0.05)',
                    linecolor: 'rgba(255,255,255,0.1)',
                    title: 'Actual Load (MW)',
                    range: [minVal, maxVal]
                }},
                yaxis: {{
                    gridcolor: 'rgba(255,255,255,0.05)',
                    linecolor: 'rgba(255,255,255,0.1)',
                    title: 'Predicted Load (MW)',
                    range: [minVal, maxVal]
                }},
                showlegend: false
            }};
            
            Plotly.newPlot(elementId, [tracePoints, traceRefLine], layout);
        }}

        drawScatter('scatterArima', arima, '#fb923c', 'ARIMA Forecast');
        drawScatter('scatterLstm', lstm, '#38bdf8', 'LSTM Forecast');
        drawScatter('scatterXgb', xgbData, '#c084fc', 'XGBoost Forecast');


        // ------------------ PLOT 10: DECOMPOSITION (Observed, Trend, Seasonal, Resid) ------------------
        const decompData = {json.dumps(decomp_data)};

        if (decompData) {{
            const traceObserved = {{
                x: decompData.dates,
                y: decompData.observed,
                xaxis: 'x',
                yaxis: 'y1',
                name: 'Observed',
                type: 'scatter',
                mode: 'lines',
                line: {{ color: '#f8fafc', width: 1.5 }}
            }};
            
            const traceTrend = {{
                x: decompData.dates,
                y: decompData.trend,
                xaxis: 'x',
                yaxis: 'y2',
                name: 'Trend',
                type: 'scatter',
                mode: 'lines',
                line: {{ color: '#fb923c', width: 2.0 }}
            }};
            
            const traceSeasonal = {{
                x: decompData.dates,
                y: decompData.seasonal,
                xaxis: 'x',
                yaxis: 'y3',
                name: 'Daily Seasonal',
                type: 'scatter',
                mode: 'lines',
                line: {{ color: '#38bdf8', width: 1.5 }}
            }};
            
            const traceResid = {{
                x: decompData.dates,
                y: decompData.resid,
                xaxis: 'x',
                yaxis: 'y4',
                name: 'Residuals',
                type: 'scatter',
                mode: 'markers',
                marker: {{ color: '#34d399', size: 5, opacity: 0.6 }}
            }};
            
            const decompLayout = {{
                grid: {{ rows: 4, columns: 1, pattern: 'coupled' }},
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ family: 'Inter', color: '#94a3b8' }},
                margin: {{ t: 20, r: 20, l: 70, b: 40 }},
                height: 620,
                xaxis: {{
                    gridcolor: 'rgba(255,255,255,0.05)',
                    linecolor: 'rgba(255,255,255,0.1)',
                    title: 'Date & Time'
                }},
                yaxis1: {{ gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.1)', title: 'Observed' }},
                yaxis2: {{ gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.1)', title: 'Trend' }},
                yaxis3: {{ gridcolor: 'rgba(255,255,255,0.05)', linecolor: 'rgba(255,255,255,0.1)', title: 'Seasonal' }},
                yaxis4: {{ 
                    gridcolor: 'rgba(255,255,255,0.05)', 
                    linecolor: 'rgba(255,255,255,0.1)', 
                    title: 'Residuals',
                    zeroline: true,
                    zerolinecolor: '#ef4444',
                    zerolinewidth: 1
                }},
                showlegend: false
            }};
            
            Plotly.newPlot('decompPlot', [traceObserved, traceTrend, traceSeasonal, traceResid], decompLayout);
        }}
    </script>
</body>
</html>
"""
    report_path = os.path.join(REPORTS_DIR, output_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Interactive dashboard generated successfully at: {report_path}")
    return report_path

def save_static_plots(dataset_name, actual_dates, actual_values, 
                      arima_preds, lstm_preds, xgb_preds, 
                      lstm_loss_history=None, xgb_model=None, 
                      feature_cols=None, peak_threshold=None, 
                      arima_intervals=None, lstm_intervals=None, xgb_intervals=None,
                      prefix="forecast"):
    """
    Saves 10 high-quality static visualizations (PNG) to reports/plots/ directory:
    1. Out-of-sample Forecast Comparison (zoomed 14 days)
    2. Residuals Error Density Estimation
    3. LSTM Training Loss Convergence
    4. Monthly Load Distribution Boxplots
    5. Weekly Load Profiles (Day of Week)
    6. Daily Load Profiles (Hourly double-peak profile)
    7. XGBoost Feature Importance
    8. Actual vs Predicted Scatter Alignments
    9. Peak Grid Stress Confusion Matrix Panel (F1 evaluation)
    10. STL-like Seasonal Decomposition
    """
    import os  # pyrefly: ignore [missing-import]
    import numpy as np  # pyrefly: ignore [missing-import]
    import pandas as pd  # pyrefly: ignore [missing-import]
    import matplotlib  # pyrefly: ignore [missing-import]
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt  # pyrefly: ignore [missing-import]
    import seaborn as sns  # pyrefly: ignore [missing-import]
    from statsmodels.tsa.seasonal import seasonal_decompose  # pyrefly: ignore [missing-import]
    from sklearn.metrics import confusion_matrix  # pyrefly: ignore [missing-import]
    
    # Create sub-directory for plots
    plots_dir = os.path.join(REPORTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Configure custom high-end Dark Theme for Seaborn/Matplotlib
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#0f172a", 
        "figure.facecolor": "#0b0f19",
        "text.color": "#f8fafc", 
        "axes.labelcolor": "#94a3b8",
        "xtick.color": "#94a3b8", 
        "ytick.color": "#94a3b8",
        "grid.color": "#1e293b",
        "axes.edgecolor": "#1e293b"
    })
    
    # Convert dates to pandas datetime
    dates_pd = pd.to_datetime(actual_dates)
    y_true = np.array(actual_values)
    y_arima = np.array(arima_preds)
    y_lstm = np.array(lstm_preds)
    y_xgb = np.array(xgb_preds)
    
    # Residuals
    res_arima = y_true - y_arima
    res_lstm = y_true - y_lstm
    res_xgb = y_true - y_xgb
    
    # Default peak threshold to 90th percentile of test actuals if not provided
    if peak_threshold is None:
        peak_threshold = float(np.percentile(y_true, 90))

    # --- PLOT 1: Forecast vs Actuals Zoomed View (Last 14 Days / 336 Hours) ---
    zoom = 336
    plt.figure(figsize=(14, 6))
    
    # Fill prediction interval bands first so they sit in the background
    if arima_intervals is not None:
        plt.fill_between(dates_pd[-zoom:], arima_intervals[0][-zoom:], arima_intervals[1][-zoom:], 
                         color="#fb923c", alpha=0.12, label="ARIMA 95% PI")
    if lstm_intervals is not None:
        plt.fill_between(dates_pd[-zoom:], lstm_intervals[0][-zoom:], lstm_intervals[1][-zoom:], 
                         color="#38bdf8", alpha=0.12, label="LSTM 95% PI")
    if xgb_intervals is not None:
        plt.fill_between(dates_pd[-zoom:], xgb_intervals[0][-zoom:], xgb_intervals[1][-zoom:], 
                         color="#c084fc", alpha=0.12, label="XGBoost 95% PI")

    plt.plot(dates_pd[-zoom:], y_true[-zoom:], label="Actual Demand", color="#f8fafc", linewidth=2.0)
    plt.plot(dates_pd[-zoom:], y_arima[-zoom:], label="ARIMA Forecast", color="#fb923c", linewidth=1.5, linestyle=":")
    plt.plot(dates_pd[-zoom:], y_lstm[-zoom:], label="LSTM Forecast", color="#38bdf8", linewidth=1.8)
    plt.plot(dates_pd[-zoom:], y_xgb[-zoom:], label="XGBoost Forecast", color="#c084fc", linewidth=1.5, linestyle="--")
    plt.title(f"1. Electricity Demand Forecast vs Actuals (Zoomed Last 14 Days)", fontsize=13, fontweight="bold", color="#fff", pad=15)
    plt.xlabel("Timestamp", fontsize=10)
    plt.ylabel("Demand (MW)", fontsize=10)
    plt.legend(facecolor="#0f172a", edgecolor="#1e293b", labelcolor="#f8fafc")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{prefix}_1_comparison.png"), dpi=120)
    plt.close()

    # --- PLOT 2: Residuals Error Density Estimation ---
    plt.figure(figsize=(10, 5))
    sns.kdeplot(res_arima, label=f"ARIMA (Mean Error: {res_arima.mean():.1f} MW)", color="#fb923c", fill=True, alpha=0.2)
    sns.kdeplot(res_lstm, label=f"LSTM (Mean Error: {res_lstm.mean():.1f} MW)", color="#38bdf8", fill=True, alpha=0.2)
    sns.kdeplot(res_xgb, label=f"XGBoost (Mean Error: {res_xgb.mean():.1f} MW)", color="#c084fc", fill=True, alpha=0.2)
    plt.axvline(0, color="#ef4444", linestyle="--", linewidth=1.5, label="Perfect Forecast")
    plt.title("2. Forecast Error Residuals Density Estimation", fontsize=13, fontweight="bold", color="#fff", pad=15)
    plt.xlabel("Error (Actual - Predicted) MW", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend(facecolor="#0f172a", edgecolor="#1e293b", labelcolor="#f8fafc")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{prefix}_2_residuals.png"), dpi=120)
    plt.close()

    # --- PLOT 3: LSTM Loss History ---
    plt.figure(figsize=(10, 5))
    if lstm_loss_history and 'loss' in lstm_loss_history:
        epochs = range(1, len(lstm_loss_history['loss']) + 1)
        plt.plot(epochs, lstm_loss_history['loss'], label="Train Loss (MSE)", color="#a78bfa", marker="o", linewidth=2)
        if 'val_loss' in lstm_loss_history:
            plt.plot(epochs, lstm_loss_history['val_loss'], label="Validation Loss (MSE)", color="#34d399", marker="s", linewidth=2)
        plt.title("3. LSTM Neural Network Loss Convergence", fontsize=13, fontweight="bold", color="#fff", pad=15)
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Loss (MSE)", fontsize=10)
        plt.legend(facecolor="#0f172a", edgecolor="#1e293b", labelcolor="#f8fafc")
    else:
        plt.text(0.5, 0.5, "LSTM Loss History Not Available (Run in Full Mode)", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12, color="#fb923c")
        plt.title("3. LSTM Loss Curve (Placeholder)", fontsize=13, fontweight="bold", color="#fff", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{prefix}_3_lstm_loss.png"), dpi=120)
    plt.close()

    # Create temporary pandas dataframe from test set to generate seasonal boxplots/profiles
    temp_df = pd.DataFrame({'Datetime': dates_pd, 'demand': y_true})
    temp_df['month'] = temp_df['Datetime'].dt.month
    temp_df['dayofweek'] = temp_df['Datetime'].dt.dayofweek
    temp_df['hour'] = temp_df['Datetime'].dt.hour
    
    # --- PLOT 4: Monthly Load Distribution Boxplots ---
    plt.figure(figsize=(12, 6))
    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    temp_df['month_name'] = temp_df['month'].map(month_names)
    sns.boxplot(x='month_name', y='demand', data=temp_df, 
                order=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
                palette="flare")
    plt.title("4. Monthly Demand Distribution (Boxplot)", fontsize=13, fontweight="bold", color="#fff", pad=15)
    plt.xlabel("Month", fontsize=10)
    plt.ylabel("Demand (MW)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{prefix}_4_monthly_boxplot.png"), dpi=120)
    plt.close()

    # --- PLOT 5: Weekly Load Profiles (Day of Week) ---
    plt.figure(figsize=(10, 5))
    day_names = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    temp_df['day_name'] = temp_df['dayofweek'].map(day_names)
    day_means = temp_df.groupby('dayofweek')['demand'].mean().reset_index()
    day_means['day_name'] = day_means['dayofweek'].map(day_names)
    sns.barplot(x='day_name', y='demand', data=day_means, palette="mako")
    plt.title("5. Average Demand by Day of Week", fontsize=13, fontweight="bold", color="#fff", pad=15)
    plt.xlabel("Day of Week", fontsize=10)
    plt.ylabel("Average Load (MW)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{prefix}_5_weekly_profile.png"), dpi=120)
    plt.close()

    # --- PLOT 6: Daily Load Profiles (Hourly Profile) ---
    plt.figure(figsize=(10, 5))
    # Plot weekday vs weekend average hourly profile
    temp_df['is_weekend'] = temp_df['dayofweek'].map(lambda d: "Weekend" if d >= 5 else "Weekday")
    hourly_profile = temp_df.groupby(['is_weekend', 'hour'])['demand'].mean().reset_index()
    sns.lineplot(x='hour', y='demand', hue='is_weekend', data=hourly_profile, palette=["#38bdf8", "#fb923c"], linewidth=2.5)
    plt.title("6. Average Hourly Demand Profile (Weekday vs Weekend)", fontsize=13, fontweight="bold", color="#fff", pad=15)
    plt.xlabel("Hour of Day", fontsize=10)
    plt.ylabel("Demand (MW)", fontsize=10)
    plt.xticks(range(0, 24, 2))
    plt.legend(facecolor="#0f172a", edgecolor="#1e293b", labelcolor="#f8fafc")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{prefix}_6_hourly_profile.png"), dpi=120)
    plt.close()

    # --- PLOT 7: XGBoost Feature Importance ---
    plt.figure(figsize=(10, 6))
    if xgb_model is not None and feature_cols is not None:
        importances = xgb_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(12, len(feature_cols))
        
        top_importances = importances[indices[:top_n]]
        top_features = [feature_cols[i] for i in indices[:top_n]]
        
        sns.barplot(x=top_importances, y=top_features, palette="viridis")
        plt.title("7. XGBoost Top Feature Importance", fontsize=13, fontweight="bold", color="#fff", pad=15)
        plt.xlabel("Relative Importance Score", fontsize=10)
        plt.ylabel("Feature", fontsize=10)
    else:
        plt.text(0.5, 0.5, "XGBoost Model Feature Importances Not Found", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12, color="#94a3b8")
        plt.title("7. XGBoost Feature Importance (Placeholder)", fontsize=13, fontweight="bold", color="#fff", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{prefix}_7_feature_importance.png"), dpi=120)
    plt.close()

    # --- PLOT 8: Actual vs Predicted Scatter Grid ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    fig.patch.set_facecolor("#0b0f19")
    
    models_dict = {
        'ARIMA Forecast': (y_arima, '#fb923c', axes[0]),
        'LSTM Forecast': (y_lstm, '#38bdf8', axes[1]),
        'XGBoost Forecast': (y_xgb, '#c084fc', axes[2])
    }
    
    # Reference line boundary
    min_val = min(y_true.min(), y_arima.min(), y_lstm.min(), y_xgb.min()) * 0.95
    max_val = max(y_true.max(), y_arima.max(), y_lstm.max(), y_xgb.max()) * 1.05
    
    for name, (preds, color, ax) in models_dict.items():
        ax.set_facecolor("#0f172a")
        ax.scatter(y_true, preds, alpha=0.3, color=color, edgecolors='none', s=8)
        ax.plot([min_val, max_val], [min_val, max_val], color='#ef4444', linestyle='--', linewidth=1.5)
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_title(name, fontsize=12, fontweight="bold", color="#fff")
        ax.set_xlabel("Actual Demand (MW)", fontsize=10, color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.grid(color="#1e293b")
        
    axes[0].set_ylabel("Predicted Demand (MW)", fontsize=10, color="#94a3b8")
    plt.suptitle("8. Actual vs Predicted Scatter Alignment (diagonal is y=x)", fontsize=14, fontweight="bold", color="#fff", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{prefix}_8_scatter_grid.png"), dpi=120, facecolor="#0b0f19")
    plt.close()

    # --- PLOT 9: Peak Stress Confusion Matrix Panel ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0b0f19")
    
    y_true_bin = (y_true > peak_threshold).astype(int)
    
    models_bin = {
        'ARIMA Peaks': (y_arima > peak_threshold).astype(int),
        'LSTM Peaks': (y_lstm > peak_threshold).astype(int),
        'XGBoost Peaks': (y_xgb > peak_threshold).astype(int)
    }
    
    for i, (name, preds_bin) in enumerate(models_bin.items()):
        cm = confusion_matrix(y_true_bin, preds_bin)
        # Plot confusion matrix heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    annot_kws={"size": 14, "weight": "bold"},
                    xticklabels=['Normal', 'Peak Event'], yticklabels=['Normal', 'Peak Event'])
        axes[i].set_title(name, fontsize=12, fontweight="bold", color="#fff")
        axes[i].set_xlabel("Predicted Label", fontsize=10, color="#94a3b8")
        axes[i].tick_params(colors="#94a3b8")
        
    axes[0].set_ylabel("Actual Label", fontsize=10, color="#94a3b8")
    plt.suptitle(f"9. Peak Demand Event Detection Confusion Matrix (Threshold: {peak_threshold:.1f} MW)", fontsize=14, fontweight="bold", color="#fff", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{prefix}_9_confusion_matrices.png"), dpi=120, facecolor="#0b0f19")
    plt.close()

    # --- PLOT 10: STL-like Seasonal Decomposition (Daily Cycle: Period = 24) ---
    # Downsample slightly to make it readable (take the last 168 hours = 1 week of data)
    decomp_subset = temp_df.iloc[-168:].set_index('Datetime')
    
    try:
        decomposition = seasonal_decompose(decomp_subset['demand'], model='additive', period=24)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
        fig.patch.set_facecolor("#0b0f19")
        
        # Color definitions
        c_obs, c_trend, c_seas, c_resid = "#f8fafc", "#fb923c", "#38bdf8", "#34d399"
        
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor("#0f172a")
            ax.tick_params(colors="#94a3b8")
            ax.grid(color="#1e293b")
            
        ax1.plot(decomposition.observed, color=c_obs, linewidth=1.5)
        ax1.set_ylabel("Observed", fontsize=10, color="#94a3b8")
        ax1.set_title("10. Additive Time Series Decomposition (Last 7 Days)", fontsize=13, fontweight="bold", color="#fff", pad=10)
        
        ax2.plot(decomposition.trend, color=c_trend, linewidth=2.0)
        ax2.set_ylabel("Trend", fontsize=10, color="#94a3b8")
        
        ax3.plot(decomposition.seasonal, color=c_seas, linewidth=1.5)
        ax3.set_ylabel("Daily Seasonal", fontsize=10, color="#94a3b8")
        
        ax4.scatter(decomposition.resid.index, decomposition.resid, color=c_resid, alpha=0.6, s=12)
        ax4.axhline(0, color="#ef4444", linestyle="--", linewidth=1)
        ax4.set_ylabel("Residuals", fontsize=10, color="#94a3b8")
        ax4.set_xlabel("Date", fontsize=10, color="#94a3b8")
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{prefix}_10_decomposition.png"), dpi=120, facecolor="#0b0f19")
        plt.close()
        print("Successfully generated all 10 diagnostic plots.")
    except Exception as e:
        print(f"Warning: Time-series decomposition failed ({e}). Saving placeholder plot.")
        plt.figure(figsize=(10, 5))
        plt.text(0.5, 0.5, f"Seasonal Decomposition Failed:\n{e}", horizontalalignment='center', verticalalignment='center', color="#ef4444")
        plt.savefig(os.path.join(plots_dir, f"{prefix}_10_decomposition.png"), dpi=120, facecolor="#0b0f19")
        plt.close()
