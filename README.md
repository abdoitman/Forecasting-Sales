# Sales Forecasting: Time Series Analysis and Prediction

## Overview

This project implements a comprehensive **time series analysis and forecasting pipeline** for multi-product, multi-location sales data. The analysis covers data exploration, preprocessing, outlier detection, feature engineering, and time series decomposition to build predictive models for sales quantity forecasting across different cities and product categories.

## Project Objectives

- **Data Exploration & Analysis (EDA)**: Understand patterns, correlations, and distributions in sales data
- **Data Cleaning**: Handle missing values, outliers, and data quality issues
- **Feature Engineering**: Create temporal features and extract time series characteristics
- **Stationarity Testing**: Validate assumptions for time series modeling
- **Seasonal Decomposition**: Identify and analyze seasonal patterns
- **Sales Forecasting**: Build predictive models for inventory and demand planning

## Dataset Overview

### Data Source
- **File**: `projectdata.xlsx - sheet 1.csv`
- **Time Period**: 2021-2023
- **Geographic Coverage**: Multiple cities (North, South, East)
- **Products**: Multiple product categories (Product X, Product Y, Product Z)

### Key Features
- **date**: Transaction date/timestamp
- **city**: Geographic location of the sale
- **product_name**: Name/identifier of the product
- **quantity**: Number of units sold (target variable)
- **retail_price**: Retail selling price
- **unit_price**: Unit cost
- **discount**: Applied discount percentage
- **area**: Sales area/region

### Data Characteristics
- **Time Granularity**: Daily 
- **Multiple Time Series**: 9 separate time series (3 cities × 3 products)
- **Missing Values**: Scattered across dates (imputed/handled)
- **Outliers**: Removed using IQR-based methodology
- **Seasonality**: 7-day seasonal pattern identified across all products
- **Data Quality**: 2021 sparse; primary analysis on 2022-2023

## Methodology

### 1. Data Preprocessing

#### Cleaning
- Rename columns for consistency (`retail price` → `retail_price`, `UnitePrice` → `unit_price`)
- Convert date strings to datetime format with daily floor granularity
- Fix product name inconsistencies (e.g., `product x ` → `product x`)
- Handle missing values in the `area` column

#### Aggregation
- Group raw data by city, product, and date
- Sum quantities to create daily aggregates
- Remove irrelevant columns (discount, prices, area) after grouping

### 2. Exploratory Data Analysis

#### Correlation Analysis
- Compute Pearson correlation matrices for all datasets
- Create separate correlation heatmaps for each product-city combination (3×3 grid)
- **Key Finding**: Prices and discounts show **no correlation** with sales quantity
- **Implication**: Suggests price-inelastic demand in this market

#### Distribution Analysis
- Box plots to identify range and potential outliers
- Histograms to visualize distribution shapes across bins
- Statistical summaries (mean, std, min, max, quartiles)
- Identify products/cities with extreme or unusual distributions

### 3. Outlier Detection & Removal

**Method**: Interquartile Range (IQR) based filtering

**Formula**:
```
IQR = Q3 - Q1
lower_bound = max(Q1 - k × IQR, 0)
upper_bound = Q3 + k × IQR
```

**Configuration**:
- **k-value used**: 1.25 (customizable parameter)
- **Applied per**: Each product-city combination independently
- **Result**: Removed approximately 3% of data points
- **Outcome**: Cleaner, more normally-distributed data for modeling

**Process**:
- Identify bounds for each product-city pair
- Flag and count outliers
- Generate per-pair statistics
- Concatenate cleaned subsets

### 4. Data Validation & Filtering

- **Year Filtering**: Excluded 2021 due to sparse data coverage
- **Justification**: 2021 underrepresented; insufficient for reliable patterns
- **Final Dataset**: Only 2022-2023 data retained
- **Impact**: 9 complete, high-quality time series for modeling

### 5. Time Series Components Analysis

#### Missing Value Patterns
- Visualized data gaps using vertical red lines overlaid on time series plots
- Identified products/cities with discontinuities
- Quantified percentage of missing values per product-city pair
- Guided interpolation strategy selection

#### Stationarity Testing
- **Method**: Augmented Dickey-Fuller (ADF) test
- **Null Hypothesis**: Time series has unit root (non-stationary)
- **Significance Level**: α = 0.05
- **Interpretation**: 
  - p-value < 0.05: Reject null → **Stationary**
  - p-value ≥ 0.05: Fail to reject → **Non-Stationary**
- **Results**: Most time series are **non-stationary** (requires differencing)
  - Exception: Product Y in North region shows stationarity

#### Seasonal Decomposition
- **Method**: Additive seasonal decomposition
- **Formula**: Y(t) = Trend(t) + Seasonal(t) + Residual(t)
- **Missing Value Handling**: Linear time-based interpolation before decomposition
- **All Series**: Identify 7-day seasonal cycles
- **Components Extracted**:
  - **Trend**: Long-term directional movement (smoothed)
  - **Seasonal**: Repeating 7-day patterns (same magnitude each week)
  - **Residual**: Irregular variations after trend and seasonal removal
- **Interpretation**: Weekly cycles likely driven by shopping behavior (mid-week peaks, weekend patterns)

### 6. Feature Engineering

#### Temporal Features
- **day**: Day of month (1-31) - captures monthly patterns
- **month**: Month of year (1-12) - captures seasonal trends
- **day_of_week**: Day in week (0=Monday, 6=Sunday) - captures weekly patterns
- **day_of_year**: Day in year (1-365) - captures annual cycles
- **day_sin / day_cos**: Cyclic encoding of day-of-week using sine/cosine transformation
  - Formula: `sin = sin(2π × day_of_week / 7)`, `cos = cos(2π × day_of_week / 7)`
  - **Purpose**: Captures circularity (Sunday→Monday transition)
  - **Advantage**: Treats day-of-week as cyclic, not linear
- **is_weekend**: Binary indicator (1 for Fri-Sat, 0 otherwise) - separates weekend effects
- **gap**: Days elapsed since previous non-null observation - identifies drought periods

#### Categorical Encoding
- **city**: Categorical variable for location (3 unique values)
- **product_name**: Categorical variable for product type (3 unique values)
- **Encoding Type**: Pandas categorical dtype (efficient storage and future one-hot encoding ready)

## Notebook Structure

The `TimeSeriesAnalysis.ipynb` notebook is organized into the following logical sections:

### 1. Load Data
- Import required libraries (pandas, numpy, matplotlib, seaborn)
- Read and parse CSV data
- Initial column renaming and datetime conversion

### 2. Data Exploration
- Correlation analysis with heatmaps
- Distribution and pricing analysis
- Identify unique areas and geographic patterns
- Product-city frequency analysis

### 3. EDA (Exploratory Data Analysis)
- Aggregate data by city, product, and date
- Generate descriptive statistics
- Box plots and histograms of quantity distribution

### 4. Handling Outliers
- Implement IQR-based outlier detection function
- Apply custom k-parameter (1.25)
- Compare before/after data shapes
- Visualize cleaned distributions

### 5. Data Validation
- Visualize temporal patterns per city
- Plot original time series with gap identification
- Year-based filtering (retain 2022-2023)
- Document missing value percentages

### 6. Stationarity Testing
- Implement ADF test wrapper function
- Test each product-city combination
- Report stationary vs. non-stationary classifications
- Document p-values for non-stationary series

### 7. Seasonal Decomposition
- Implement seasonal decomposition function
- Extract 7-day seasonality information
- Visualize seasonal components
- Confirm consistent seasonal period across products

### 8. Data Interpolation
- Implement interpolation logic (zeros or linear methods)
- Track originally-missing values with flag column
- Support both methods for comparison

### 9. Feature Engineering
- Create temporal feature extraction function
- Generate all derived features from timestamps
- Encode categorical variables
- Sort and finalize feature set

## Key Findings

### 1. Sales Price Independence
- **Finding**: Retail and unit prices show no correlation with quantity sold
- **Interpretation**: Suggests price-inelastic demand or market saturation
- **Business Implication**: Pricing alone unlikely to drive volume; focus on availability/convenience

### 2. Strong Weekly Seasonality
- **Pattern**: All products exhibit clear 7-day seasonal cycles
- **Likely Driver**: Weekly shopping patterns (e.g., mid-week peaks, weekend variations)
- **Frequency**: Consistent across all 9 product-city combinations
- **Modeling**: Enables seasonal models (SARIMA with period=7)

### 3. Non-Stationarity
- **Observation**: Most time series are non-stationary (p-value > 0.05 in ADF test)
- **Exception**: Product Y in North region shows stationarity
- **Requirement**: Must apply differencing or seasonal adjustment before ARIMA modeling
- **Approach**: Use SARIMA (seasonal ARIMA) to handle trend + seasonality

### 4. Data Quality & Completeness
- **2021 Data**: Sparse coverage; excluded from primary analysis
- **2022-2023**: Complete, reliable data with manageable gaps
- **Outliers**: ~3% removed via IQR filtering (1.25k)
- **Missing Values**: Scattered but addressable via interpolation

### 5. Missing Data Patterns
- **Nature**: Non-random gaps likely due to operational issues
- **Impact**: Manageable with interpolation; won't severely bias models
- **Handling**: Linear interpolation captures trends better than zero-filling

## Technologies & Dependencies

### Core Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| **pandas** | Latest | Data manipulation, aggregation, time series indexing |
| **numpy** | Latest | Numerical computations, IQR calculations |
| **matplotlib** | Latest | Static data visualization (plots, histograms) |
| **seaborn** | Latest | Statistical visualization (heatmaps, boxplots) |
| **statsmodels** | Latest | Time series decomposition (seasonal_decompose), ADF test |

### Installation

```bash
pip install pandas numpy matplotlib seaborn statsmodels
```

Or using conda:
```bash
conda install pandas numpy matplotlib seaborn statsmodels
```

## How to Use

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required libraries installed (see above)

### Execution Steps

1. **Navigate to Project Directory**:
   ```bash
   cd path/to/Forecasting-Sales
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook TimeSeriesAnalysis.ipynb
   ```

3. **Execute Cells Sequentially**: 
   - Run cells from top to bottom
   - Each section builds on previous results
   - Cells include markdown explanations and visualizations

4. **Customization Options**:
   - **Outlier Threshold**: Modify `percentage_of_IQR` parameter (default: 1.25)
     ```python
     outlier_free_df = drop_outliers(data_grouped, percentage_of_IQR=1.5)
     ```
   - **Interpolation Method**: Change between 'zeros' and 'linear'
     ```python
     decision = 'interpolated'
     interpolation_method = 'linear'
     ```
   - **Date Filtering**: Adjust year range in data query
     ```python
     cleaned_data = outlier_free_df.query('date.dt.year >= 2022')
     ```

### Key Outputs
- **Correlation Heatmaps**: PNG images showing feature relationships
- **Time Series Plots**: Visualizations with gap identification
- **Decomposition Components**: Trend, seasonal, and residual breakdowns
- **Feature Dataset**: Engineered features ready for modeling

## Potential Next Steps

### 1. Time Series Forecasting Models

#### ARIMA/SARIMA
```python
from statsmodels.tsa.arima.model import ARIMA
# Fit SARIMA(1,1,1)×(1,1,1)7 for 7-day seasonality
```
- **Best for**: Univariate forecasting with trend and seasonality
- **Advantage**: Interpretable, well-established theory
- **Consideration**: Requires stationarity (differencing)

#### Prophet
```python
from prophet import Prophet
# Facebook's automated forecasting tool
```
- **Best for**: Quick, automated forecasts with holiday effects
- **Advantage**: Handles missing data natively
- **Consideration**: Less interpretable than ARIMA

#### ARIMAX
- **Enhancement**: Include exogenous variables (e.g., promotions, events)
- **Benefit**: Capture external influences on quantity

### 2. Machine Learning Approaches

#### Deep Learning (LSTM/GRU)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
```
- **Best for**: Complex non-linear patterns, high-dimensional data
- **Advantage**: Captures long-range dependencies
- **Consideration**: Requires more data and hyperparameter tuning

#### Gradient Boosting (XGBoost/LightGBM)
```python
import xgboost as xgb
# Lag-based features + temporal features
```
- **Best for**: Fast training, feature importance insights
- **Advantage**: Handles non-linear relationships, robust to outliers
- **Consideration**: Requires careful lag engineering

#### Ensemble Methods
- Combine ARIMA, Prophet, LSTM, and XGBoost predictions
- Use weighted averaging or stacking
- Often achieves superior performance

### 3. Model Evaluation & Validation

#### Cross-Validation Strategy
- **Time Series Split**: Maintain temporal order (not random shuffle)
- **Walk-Forward Validation**: Simulate real-world incremental retraining
- **Avoid Data Leakage**: Test set always after training set temporally

#### Performance Metrics
- **MAE** (Mean Absolute Error): Average absolute deviation
- **RMSE** (Root Mean Squared Error): Penalizes large errors more
- **MAPE** (Mean Absolute Percentage Error): Percentage-based error
- **SMAPE** (Symmetric MAPE): More symmetric variant

#### Code Example
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

### 4. Production Deployment

#### Pipeline Architecture
```python
class SalesForecaster:
    def __init__(self, model_type='sarima'):
        self.model = None
        self.scaler = None
    
    def fit(self, train_data):
        # Train model
        pass
    
    def predict(self, horizon):
        # Generate forecasts
        pass
```

#### API Development
- Build Flask/FastAPI endpoint for predictions
- Accept city, product, horizon as inputs
- Return point forecasts + confidence intervals

#### Automated Retraining
- Schedule weekly/monthly retraining
- Monitor forecast accuracy
- Alert on performance degradation

#### Monitoring & Logging
- Track prediction errors over time
- Log data quality metrics
- Monitor data drift (distribution changes)

## Project Structure

```
Forecasting-Sales/
├── README.md                           # This file
├── LICENSE                             # Project license
├── TimeSeriesAnalysis.ipynb            # Main analysis notebook
└── projectdata.xlsx - sheet 1.csv      # Raw data file (not included)
```

## Key Insights Summary

| Aspect | Finding | Impact |
|--------|---------|--------|
| **Correlation** | Price ↔ Quantity: None | Focus forecasting on temporal patterns, not pricing |
| **Seasonality** | 7-day cycles | Use SARIMA(×7) or similar seasonal model |
| **Stationarity** | Mostly non-stationary | Apply differencing or seasonal adjustment |
| **Data Period** | 2022-2023 valid | Exclude 2021 from training; limited historical patterns |
| **Outliers** | ~3% removed | IQR filtering (k=1.25) effective for this data |
| **Missing Data** | Scattered gaps | Linear interpolation suitable for handling |

## Best Practices Applied

- **Data Cleaning**: Standardized column names, consistent date formats  
- **Outlier Handling**: Statistical method (IQR) with documentation  
- **Feature Engineering**: Domain-aware temporal features (cyclical encoding, weekend flag)  
- **Exploratory Analysis**: Multiple visualizations, statistical tests (ADF)  
- **Seasonality Identification**: Formal decomposition with component extraction  
- **Modular Code**: Reusable functions for reproducibility  
- **Documentation**: Comprehensive docstrings and inline comments  
