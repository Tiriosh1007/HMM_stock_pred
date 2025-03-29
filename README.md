# Hidden Markov Model for Stock Price Prediction

This project replicates the stock price prediction model using Hidden Markov Models as described in the following academic article:

Nguyen, N. (2017). An analysis and implementation of the hidden Markov model to technology stock prediction. Risks (Basel), 5(4), 1–16. https://doi.org/10.3390/risks5040062

## Project Overview

The implementation analyzes stock price data for major technology companies (Facebook/Meta, Apple, Google, Tesla) and the S&P500 ETF (SPY) from 2015 to 2024. The analysis includes:

1. Data retrieval and preprocessing
2. Hidden Markov Model implementation with dynamic calibration
3. Model evaluation using multiple metrics
4. Trading simulation and performance analysis

## Implementation Details

The project uses a `HMMStockTrader` class that implements the following key features:

- **Dynamic Calibration**: Trains HMM models on rolling windows of historical data
- **Model Selection**: Uses BIC (Bayesian Information Criterion) to select the best model
- **Prediction Methods**: 
  - Naive forecast
  - Best candidate HMM prediction
  - Voting-based HMM prediction
- **Evaluation Metrics**:
  - Regression metrics (R², MAPE)
  - Classification metrics (Accuracy, Precision, Recall, F1, AUC)
  - Trading performance metrics

## Getting Started

### Prerequisites
- Python 3.11 or later
- Required packages listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/Tiriosh1007/HMM_stock_pred.git
cd HMM_stock_pred

# Install required packages
pip install -r requirements.txt
```

### Usage

The main script `hmm_trading_v1.py` contains the implementation. To use it:

1. Import the `HMMStockTrader` class
2. Initialize with your desired parameters:
   ```python
   trader = HMMStockTrader(
       ticker='AAPL',
       stock_name='Apple Inc.',
       start_date='2015-01-01',
       end_date='2024-12-31',
       test_start_date='2023-01-01'
   )
   ```
3. Run the analysis:
   ```python
   results = trader.run()
   ```

## Model Parameters

The HMM implementation includes several customizable parameters:

- `train_window_size`: Size of training windows (default: 100)
- `test_window_size`: Size of test windows (default: 100)
- `sim_threshold`: Similarity threshold for model selection (default: 0.99)
- `further_calibrate`: Whether to calibrate models further (default: True)
- `em_iterations`: Number of EM iterations (default: 1)
- `look_back_period`: Period for look-back analysis (default: -1)
- `training_mode`: Mode of training (0: dynamic, 2: static)
- `n_shares`: Number of shares for trading simulation (default: 100)

## Results

The model provides:
- Price predictions
- Trading signals
- Performance metrics
- Visualization of results including:
  - Actual vs. predicted prices
  - AIC/BIC values over training windows
  - Cumulative profit/loss
