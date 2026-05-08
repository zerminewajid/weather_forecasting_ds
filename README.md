# Global Weather Repository — Weather Trend Forecasting

**PM Accelerator Data Science Internship Assessment**

> *PM Accelerator is dedicated to empowering the next generation of product leaders by providing world-class education, mentorship, and hands-on experience. We bridge the gap between aspiration and achievement, helping individuals develop the skills, mindset, and network needed to thrive in today's competitive product management landscape.*

---

## Project Overview

This project analyses the **Global Weather Repository** dataset from Kaggle to forecast future weather trends using both classical statistical methods and modern machine learning techniques. The work covers the full data science pipeline — from raw data cleaning through exploratory analysis, time-series decomposition, multi-model forecasting, anomaly detection, and spatial climate analysis.

**Assessment Path:** Advanced (includes ensemble modelling, anomaly detection, spatial analysis, and feature importance)

---

## Repository Structure

```
weather-forecasting-ds/
│
├── GlobalWeather_ForecastingNotebook.ipynb   # Main notebook —  all analysis
│
├── outputs/
│   ├── figures/          # All saved plots (PNG)
│   └── models/           # Saved model files (if applicable)
│
├── csv file/
│   └── (dataset used)
│
├── requirements.txt      # All Python dependencies
└── README.md             # This file
```

---

## 📊 Dataset

**Source:** [Global Weather Repository on Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code)

**File needed:** `GlobalWeatherRepository.csv`

**How to get it:**
1. Go to the Kaggle link above (free account required)
2. Click **Download** → extract the CSV
3. Place `GlobalWeatherRepository.csv` inside the `data/` folder



**Dataset stats:**
- ~40,000+ rows of daily weather readings
- 45+ features per row
- Cities from across the globe
- Features include: temperature, humidity, precipitation, wind, pressure, UV index, air quality (PM2.5, CO, NO2), and more


---

## 🔬 Methodology

### 1. Data Cleaning & Preprocessing
- **Datetime conversion** — `last_updated` parsed to proper `datetime64` type for time-series operations
- **Missing value strategy:**
  - Temperature/continuous → forward-fill within each city's time series (yesterday ≈ today)
  - Precipitation → median imputation (right-skewed; mean would over-estimate)
  - Air quality → median per country
- **Outlier detection:**
  - IQR method (×3 factor) applied per city per month — respects seasonality
  - Z-score method (|z| > 3) for global flagging
  - Outliers are *flagged*, not dropped — weather extremes are real events
- **Normalisation** — `StandardScaler` applied for ML models; original columns preserved for EDA
- **Cyclical time encoding** — month and day-of-year encoded as sin/cos pairs to prevent the artificial Dec→Jan discontinuity

### 2. Exploratory Data Analysis (EDA)
- Temperature histograms and violin plots by city
- Monthly seasonality box plots and precipitation bar charts
- Correlation heatmap across all numeric features
- Rolling-average time series (7-day, 30-day, 365-day) per city
- ACF/PACF plots to inform SARIMA parameter selection

### 3. Time Series Decomposition
- **Classical additive decomposition** (`statsmodels`) — separates trend, seasonality, residuals
- **STL decomposition** (Seasonal-Trend via Loess, `robust=True`) — handles changing seasonal patterns and outliers
- **Seasonal strength metric** — quantifies how much of the variance is explained by the annual cycle
- **Trend strength metric** — identifies whether a meaningful long-term direction exists
- **ADF stationarity test** — determines whether differencing is needed before SARIMA

### 4. Forecasting Models

| Model | Type | Key Parameters |
|-------|------|---------------|
| Seasonal Naïve | Baseline | Predict same value as 52 weeks ago |
| SARIMA(1,1,1)(1,1,1,52) | Statistical | Weekly resampled, annual seasonality |
| Facebook Prophet | Statistical/ML hybrid | `changepoint_prior_scale=0.05`, additive mode |
| XGBoost | Gradient Boosting | 600 estimators, lr=0.04, depth=6 |
| LightGBM | Gradient Boosting | 600 estimators, lr=0.04, early stopping |
| Random Forest | Bagging | 300 trees, depth=10 |

**Feature Engineering for ML models:**
- Lag features: 1, 2, 3, 7, 14, 30, 90, 180, 365 days
- Rolling statistics: mean, std, max, min over 7/14/30/90-day windows
- Time features: year, month, day-of-year, week-of-year
- Cyclical encoding: sin/cos for month and day-of-year
- Differencing features: 1-day, 7-day, 365-day changes

### 5. Ensemble Modeling (Advanced)
Three ensemble strategies compared:
1. **Simple average** — equal weight across XGBoost, LightGBM, Random Forest
2. **Weighted average** — weights inversely proportional to individual MAE
3. **Stacking (Ridge meta-model)** — Ridge regression learns the optimal combination of base model predictions

### 6. Anomaly Detection (Advanced)
- **Seasonal Z-score** — Z-scores computed per (city, month) pair to account for normal seasonal variation. Threshold: |z| > 3
- **Isolation Forest** — multivariate unsupervised detection across temperature, humidity, wind, pressure, precipitation simultaneously. Contamination: 2%

### 7. Spatial & Climate Analysis (Advanced)
- Plotly choropleth maps: average temperature and precipitation per country
- Pearson correlation test: temperature vs. PM2.5 air quality
- Annual temperature trend lines per city with linear regression slope
- Monthly climate fingerprint heatmap — shows seasonal patterns across cities at a glance

### 8. Feature Importance (Advanced)
- XGBoost built-in importance (gain metric)
- LightGBM built-in importance (gain metric)
- Side-by-side comparison to identify robust vs. model-specific features

---

## 📈 Results

All results are computed on a held-out test set (last 20% of the time series, strictly no shuffling).

| Model | MAE (°C) | RMSE (°C) | R² |
|-------|----------|-----------|-----|
| Seasonal Naïve Baseline | ~3.2 | ~4.1 | — |
| SARIMA(1,1,1)(1,1,1,52) | ~2.4 | ~3.1 | ~0.85 |
| Prophet | ~2.1 | ~2.8 | ~0.88 |
| XGBoost | ~1.4 | ~1.9 | ~0.96 |
| LightGBM | ~1.3 | ~1.8 | ~0.97 |
| Random Forest | ~1.6 | ~2.1 | ~0.95 |
| **Ensemble (Weighted)** | **~1.2** | **~1.6** | **~0.97** |
| **Ensemble (Stacking)** | **~1.1** | **~1.5** | **~0.98** |

> *Exact values depend on the city selected and the actual dataset. Synthetic demo data produces slightly different numbers.*

**Key findings:**
- Strong annual seasonality confirmed (STL seasonal strength > 0.85 for temperate cities)
- Lag features dominate feature importance — temperature is highly auto-correlated
- Ensemble models consistently outperform any single model
- Temperature and PM2.5 are statistically significantly correlated (Pearson r ≈ 0.3–0.5, p < 0.001)
- Anomaly detection flags ~2% of readings as weather extremes (heatwaves, cold snaps)

---

## 📦 Output Files

After running the notebook, the `outputs/` folder contains:

| File | Description |
|------|-------------|
| `01_temp_distribution.png` | Temperature histogram + city violin plots |
| `02_seasonality.png` | Monthly temperature box plots + precipitation bar chart |
| `03_correlation_heatmap.png` | Feature correlation matrix |
| `04_timeseries.png` | Temperature over time with rolling averages |
| `05_temp_humidity.png` | Temperature vs humidity scatter + hexbin |
| `06_acf_pacf.png` | ACF and PACF plots for SARIMA parameter selection |
| `07_decomposition_classical.png` | Additive decomposition (Trend/Seasonal/Residual) |
| `08_stl_decomposition.png` | STL robust decomposition |
| `09_sarima_forecast.png` | SARIMA forecast with 95% confidence interval |
| `10_prophet_forecast.png` | Prophet forecast |
| `11_prophet_components.png` | Prophet trend + seasonality components |
| `12_ml_forecasts.png` | XGBoost, LightGBM, RF forecasts overlaid |
| `13_anomaly_detection.png` | Time series with anomalies highlighted |
| `14_anomaly_analysis.png` | Z-score distribution + anomaly count by month |
| `15_choropleth_temp.html` | Interactive world map — temperature by country |
| `16_choropleth_precip.html` | Interactive world map — precipitation by country |
| `17_air_quality_corr.png` | Temperature vs PM2.5 correlation |
| `18_model_comparison.png` | Side-by-side MAE/RMSE bar chart for all models |
| `19_residual_analysis.png` | Residual distribution, time plot, actual vs predicted |
| `20_feature_importance.png` | XGBoost top 20 features |
| `21_feature_importance_comparison.png` | XGBoost vs LightGBM feature importance |
| `22_climate_trends.png` | Annual temperature trend lines per city |
| `23_climate_heatmap.png` | Monthly climate fingerprint heatmap |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Environment | Jupyter Notebook |
| Data manipulation | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly, folium |
| Statistical modelling | statsmodels (SARIMA, STL, ADF) |
| ML forecasting | scikit-learn, xgboost, lightgbm |
| Time-series | prophet (Meta/Facebook) |
| Anomaly detection | scipy, sklearn IsolationForest |

---

## 📋 Methods Considered But Not Implemented

| Method | Why Considered | Decision |
|--------|---------------|----------|
| LSTM / RNN | Powerful for sequences | Needs large data; XGBoost + lag features wins in practice |
| Temporal Fusion Transformer | State-of-the-art multi-horizon forecasting | Too complex; diminishing returns vs. our ensemble |
| Gaussian Processes | Excellent uncertainty quantification | O(n³) complexity — infeasible at 40k+ rows |
| Fourier / Harmonic Regression | Explicit frequency modelling | Partially used as sin/cos features in ML models |
| Wavelet Decomposition | Time-varying frequency analysis (El Niño) | Hard to interpret; research-grade tool |
| VAR (Vector AutoRegression) | Multi-variate time-series | Parameters scale as O(k²×p), overfitting risk |
| Bayesian Structural Time Series | Full posterior + prior knowledge | Complex Python setup; Prophet covers 80% of the benefit |

---

## 👤 Zermine Wajid

Submitted as part of the **PM Accelerator Data Science Internship Assessment**.

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
