````markdown
# An Exploratory Analysis of SPY (S&P 500 ETF) Price and Volume Dynamics (2015–2025)

## Abstract
This study investigates daily price and volume data for the SPDR S&P 500 ETF Trust (`SPY`) over the ten-year period from June 24, 2015 to June 24, 2025. We test the hypothesis that raw price and volume metrics exhibit minimal direct correlation due to unquantified qualitative drivers (macroeconomic shifts, policy changes, behavioral factors). Although we expect weak Pearson correlations, secondary patterns—such as asymmetries in returns at extreme volume thresholds—may yield actionable insights.

## Introduction
Financial time series often appear dominated by noise. By examining SPY’s daily observations, we aim to:

- Quantify the volume–price correlation structure.
- Identify anomalies or systematic patterns at different volume percentiles.
- Explore whether extreme volume days consistently produce negative or positive returns.

## Hypothesis
> **H₀:** Daily price and volume metrics for SPY over the last ten years are uncorrelated (ρ ≈ 0).
>
> **H₁:** Secondary effects at distribution tails (e.g., top 5% volume days) will reveal statistically significant deviations in average returns.

## Setup & Imports
```python
# Data handling
import pandas as pd
import numpy as np

# Statistical tests & modeling
from scipy.stats import pearsonr, ttest_1samp
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# Visualization
import matplotlib.pyplot as plt
````

## Data Cleaning & Feature Engineering

1. **Load & parse** raw CSV (`Date`,`Open`,`High`,`Low`,`Close/Last`,`Volume`).
2. **Sort** by date; **round** prices to two decimals; **drop** NAs.
3. **Engineered columns**:

   * `MA90`: 90-day moving average of closing price.
   * `% Change`: daily percent price change.
   * `Price Action`: qualitative tag ("increasing" / "decreasing").
   * `Intraday Range`: `High` − `Low`.
   * `Volume Difference` & `Volume Trend` (+1/−1).
   * `Volume Percentile`: rank-based percentile over full sample.

```python
spy = pd.read_csv("spydata.csv", parse_dates=["Date"])  
spy.sort_values("Date", inplace=True)  
# ... rounding and feature code ...
```

## Exploratory Data Analysis

### 1. Total Growth

* **Low**: 209.77 → 603.41
* **High**: 211.25 → 607.85
* **Total % Growth**: \~187.7% (≈ 18.77% annual arithmetic).

### 2. Volume–Price Correlation

```python
corr_high = df2["vol_pct"].corr(df2["high_pct"])  # r ≈ -0.115
corr_low  = df2["vol_pct"].corr(df2["low_pct"])   # r ≈ -0.306
```

Negative correlations suggest higher volume associates more strongly with downward price moves.

### 3. Correlation Matrix

We computed a heatmap of numeric features (`Close/Last`, `Volume`, `MA90`, `% Change`, `Intraday Range`, `Volume Trend`, `Volume Percentile`), ignoring correlations |ρ| < 0.15 as inconclusive.

## Statistical Testing

We performed one-sample t-tests of `% Change` for days above each volume quantile:

| Quantile | n days | Avg Return (%) | p-value |
| :------- | :----: | :------------: | :-----: |
|  5%      |   126  |     −0.737     |  0.0106 |
| 75%      |   629  |     −0.284     |  0.0002 |
| 99%      |   26   |     −1.748     |  0.0518 |

Significant negative returns at extreme volume thresholds support **H₁**.

## Regression & Modeling

1. **Exponential fit** (`y = a·e^(b·x)`) via `curve_fit`—captures tail but misfits bulk.
2. **Machine learning models** on `(volume percentile → avg return)` distribution:

   * Linear Regression (R² ≈ 0.73)
   * 2nd‑degree Polynomial (R² ≈ 0.81)
   * Random Forest (best: R² ≈ 0.99, mild overfit: CV mean R² ≈ 0.91)
   * MLP & custom exponential regressor underperformed.

```python
results_df = pd.DataFrame(results).set_index("model")
print(results_df)
```

## Results & Interpretation

* **Weak raw correlations**: ρ\_high = −0.115; ρ\_low = −0.306.
* **Volume spikes**: top 5% days yield significant negative returns (p < 0.05).
* **Modeling**: Random Forest best captures non‑linear return decay at extremes.
* **Overfitting check**: Train R² = 0.94 vs CV R² ≈ 0.91—generalizable with variance at mid‑percentiles.

## Next Steps

* Backtest a volume‑based trading strategy (e.g., short on extreme volumes).
* Integrate additional features: volatility, momentum indicators, macro factors.
* Employ rolling-window validation across market regimes.
* Combine models (ensemble) to stabilize mid‑range predictions.
* Deploy an interactive app: input volume percentile → predicted return.

## File Structure

```
SPY_Analysis/
├── data/                   # CSV data
├── notebooks/              # Jupyter analyses (cleaning → modeling)
├── src/                    # Reusable functions (loader, features, tests)
├── requirements.txt        # Environment specs
└── README.md               # This overview
```

## License
[MIT](LICENSE)


