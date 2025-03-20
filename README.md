# Hidden Markov Model for Stock Price Prediction

This project replicates the stock price prediction model using Hidden Markov Models as described in the following academic article:

Nguyen, N. (2017). An analysis and implementation of the hidden Markov model to technology stock prediction. Risks (Basel), 5(4), 1–16. https://doi.org/10.3390/risks5040062

## Project Overview

The implementation analyzes stock price data for major technology companies (Facebook/Meta, Apple, Google, Tesla) and the S&P500 ETF (SPY) from 2015 to 2024. The analysis includes:

1. Data retrieval and preprocessing
2. Autocorrelation function (ACF) analysis
3. Hidden Markov Model implementation
4. Model evaluation and prediction

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

# Run Jupyter Notebook
jupyter notebook hmm_stock_prediction.ipynb
```

## Implementation Details

The project uses the `hmmlearn` library to implement the Hidden Markov Model. The model attempts to identify hidden states in the stock market based on observable price data, and then uses these states to make predictions about future price movements.

The ACF analysis helps to identify patterns and dependencies in the time series data, which informs the HMM implementation.
