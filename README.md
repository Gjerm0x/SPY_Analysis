# SPY Analysis Project

## Overview
This project explores the trading dynamics of the SPDR S&P 500 ETF Trust (ticker `SPY`) over the past decade. It provides a structured analysis of price and volume data to identify patterns, anomalies, and potential trading insights. The core objectives are:

- To examine the relationship between trading volume and price movements.
- To test hypotheses about volume spikes and their impact on returns.
- To visualize key metrics and highlight statistically significant findings.

## Data Sources
- **Daily SPY Data**: Historical price and volume data for `SPY`, sourced from nasdaq and stored as CSV files.
- https://www.nasdaq.com/market-activity/etf/spy/historical

## Methodology
1. **Data Ingestion & Cleaning**
   - Load CSV data into Pandas DataFrames.
   - Parse dates, sort chronologically, and round numeric values.
2. **Feature Engineering**
   - Compute daily percentage returns (`% Change`).
   - Calculate volume percentiles and identify high-volume days.
   - Create moving averages (e.g., 90-day MA) for price.
3. **Statistical Analysis**
   - Perform t‑tests to compare average returns on high-volume days vs. normal days.
   - Run OLS regressions to quantify the strength of volume–return relationships.
4. **Visualization**
   - Generate scatter plots, and regression charts using Matplotlib.
   - Annotate charts to emphasize anomalies and key thresholds.

## Results Summary
- **Volume Spike Returns**: Days in the top 5th percentile of volume show a statistically significant difference in average returns (p < 0.05).
- **Regression Analysis**: OLS models indicate a moderate correlation between volume anomalies and daily returns (R² ≈ 0.65 for training data).
- **Anomalies**: Subtle patterns emerge around extreme volumes

## Dependencies
- Python 3.8+
- Pandas
- numpy
- matplotlib
- scipy
- statsmodels
