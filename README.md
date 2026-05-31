# Household Energy Consumption Forecasting

## Project Overview

This project focuses on forecasting daily household energy consumption using historical electricity usage data. The objective is to investigate how feature engineering, machine learning models, and time-series validation techniques can be applied to a real-world forecasting problem.

The target variable is:

```text
Global_active_power
```

which represents the total household power consumption.

---

## Dataset

Dataset:

**Individual Household Electric Power Consumption**

The dataset contains measurements collected at one-minute intervals, including:

| Feature | Description |
|----------|----------|
| Global_active_power | Household active power consumption |
| Global_reactive_power | Household reactive power consumption |
| Voltage | Voltage measurements |
| Global_intensity | Current intensity |
| Sub_metering_1 | Kitchen-related energy consumption |
| Sub_metering_2 | Laundry-related energy consumption |
| Sub_metering_3 | Climate control and water-heater consumption |

---

## Data Preprocessing

### Missing Values

The original dataset contains missing values represented by:

```text
?
```

These values were converted to NaN during data loading.

### Datetime Index

The Date and Time columns were combined into a single datetime index, enabling time-based operations and forecasting workflows.

### Interpolation

Missing values were filled using:

```python
interpolate(method="time")
```

This method estimates missing observations using neighboring timestamps while preserving temporal continuity.

---

## Resampling

The original dataset contains minute-level measurements.

To simplify the forecasting task and reduce noise, the data was resampled to daily frequency:

```python
df_daily = df.resample("D").mean()
```

This converts all observations within a day into a single daily average value.

Benefits:

- Reduced noise
- Faster training
- Easier visualization
- More interpretable forecasting results

---

## Exploratory Data Analysis

Several observations were made during visualization:

- Energy consumption exhibits seasonal patterns.
- Consumption varies across different periods of the year.
- Strong relationships exist between electrical measurements and power consumption.

### Correlation Analysis

A correlation heatmap revealed strong associations between:

- Global_active_power
- Global_intensity
- Sub_metering variables

These findings guided later feature-engineering decisions.

---

## Feature Engineering

### Lag Features

Historical consumption information was incorporated using:

```text
lag_1
lag_7
lag_14
lag_30
```

These features provide:

- Short-term memory
- Weekly seasonality
- Monthly patterns

### Rolling Statistics

Rolling-window features were created to capture local trends and variability:

```text
rolling_mean_7
rolling_mean_30
rolling_std_7
```

These summarize recent behavior and help the model identify trends.

### Calendar Features

Seasonal indicators were added:

```text
day_of_week
month
```

These features help the model learn repeating weekly and monthly patterns.

---

## Data Leakage Investigation

An early forecasting model achieved unrealistically strong performance.

Investigation revealed that same-day electrical measurements were being used to predict same-day power consumption.

Examples:

```text
Global_intensity
Sub_metering_1
Sub_metering_2
Sub_metering_3
```

Since these values would not be available when forecasting future consumption, they were removed from the final forecasting model.

Only historical information was retained.

This produced a more realistic forecasting setup.

---

## Models Evaluated

### Linear Regression

A simple baseline model was used to establish benchmark performance.

Strengths:

- Fast
- Interpretable
- Effective for smooth relationships

### Random Forest Regressor

A tree-based ensemble model was evaluated to capture nonlinear patterns.

Observation:

Random Forest did not outperform Linear Regression on the resampled daily data.

### XGBoost Regressor

Gradient boosting was used to improve forecasting accuracy.

Key parameters:

```python
XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
```

XGBoost achieved the best overall performance.

---

## Validation Strategy

A random train/test split is inappropriate for forecasting tasks because future information can leak into the training set.

Instead, the project uses:

```python
TimeSeriesSplit
```

This preserves chronological order and simulates real forecasting conditions.

Five validation folds were evaluated.

---

## Results

### Single Hold-Out Evaluation

Best model:

```text
XGBoost
```

Performance:

```text
MAE  ≈ 0.16
RMSE ≈ 0.23
R²   ≈ 0.58
```

### TimeSeriesSplit Cross-Validation

Average performance:

```text
MAE  ≈ 0.19
RMSE ≈ 0.25
R²   ≈ 0.51
```

The cross-validation results provide a more realistic estimate of forecasting performance across different time periods.

---

## Feature Importance

XGBoost feature importance analysis showed that the most influential predictors were:

- rolling_mean_7
- rolling_std_7
- day_of_week
- lag_1
- rolling_mean_30

These results indicate that recent trends and recent variability are highly informative for forecasting household energy consumption.

---

## Key Concepts Learned

This project demonstrates:

- Time-series preprocessing
- Datetime indexing
- Missing-value handling
- Time-based interpolation
- Resampling
- Feature engineering
- Lag features
- Rolling-window statistics
- Calendar features
- Data leakage prevention
- TimeSeriesSplit cross-validation
- XGBoost forecasting
- Feature importance analysis

---

## Future Improvements

Potential extensions include:

- Hourly forecasting
- Prophet forecasting
- ARIMA/SARIMA models
- LSTM-based forecasting
- Hyperparameter optimization
- Additional seasonal features
- Multi-step forecasting

---

## Final Observations

Linear Regression generalized surprisingly well despite its simplicity.

Random Forest did not outperform the linear baseline on daily aggregated data.

XGBoost achieved the strongest overall performance and benefited from carefully engineered lag and rolling-window features.

The project highlights that in forecasting tasks, feature engineering and leakage prevention are often more important than model complexity.
