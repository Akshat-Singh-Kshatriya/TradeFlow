# TradeFlow: AI-Driven Option Strategy

## Overview
A trading engine that uses **Machine Learning** to predict volatility regimes and dynamically deploy **Butterfly** option strategies. It includes a built-in **Black-Scholes pricing model** and **Monte Carlo risk engine** to calculate Probability of Profit (PoP) before taking a trade.

## Key Features
- **Predictive Modeling:** Uses Technical Indicators (RSI, MACD, Bollinger Bands) to forecast price consolidation.
- **Dynamic Pricing:** Calculates theoretical Option Premiums & Greeks (Delta, Gamma, Theta, Vega) in real-time.
- **Risk Analysis:** Runs 10,000 Monte Carlo simulations to estimate VaR (Value at Risk) and Win Rate.

## Tech Stack
- Python, pandas, numpy, scipy
- Scikit-Learn (Random Forest & Gradient Boosting)
- Matplotlib & Seaborn (Financial Visualization)

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the engine: `python app.py`
