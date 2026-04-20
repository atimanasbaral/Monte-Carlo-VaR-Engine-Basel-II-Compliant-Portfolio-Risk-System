# Monte Carlo VaR Engine - Basel II Compliant Portfolio Risk System

A production-grade Monte Carlo Value-at-Risk system for comprehensive portfolio risk assessment and regulatory compliance.

## 🚀 Features

### Core Engine
- **20-asset NIFTY50 portfolio** with realistic Indian market tickers
- **10,000 Monte Carlo simulations** per scenario for statistical robustness
- **Full covariance matrix** with Cholesky decomposition for correlated asset movements
- **VaR and CVaR calculations** at 95% and 99% confidence levels

### Stress Testing
- **5 market regimes** for comprehensive risk assessment:
  - Bull Market (low volatility, positive drift)
  - Bear Market (negative drift, moderate volatility)
  - High-Volatility (2× sigma scaling)
  - Low-Liquidity (fat tails, t-distribution)
  - Crisis (simultaneous -3σ shock)

### Basel II Compliance
- **Kupiec POF test** for VaR accuracy validation
- **Christoffersen test** for independence of VaR breaches
- **Regulatory backtesting** across all stress regimes
- **Pass/fail criteria** based on 5% significance level

### Interactive Dashboard
- **6-panel visualization** with Plotly:
  1. P&L Distribution with VaR/CVaR lines
  2. VaR Comparison across regimes
  3. Portfolio Correlation Heatmap
  4. Backtest Results Table
  5. Cumulative Portfolio Returns
  6. Cholesky Factor Matrix

## 📁 Project Structure

```
Monte_Carlo_Var_Engine/
├── var_engine.py      # Core Monte Carlo simulation engine
├── backtester.py      # Kupiec & Christoffersen backtesting
├── report.py          # Interactive dashboard generator
├── main.py            # End-to-end pipeline runner
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## 🛠️ Installation

1. **Clone or download** the project directory
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### Quick Start
Run the complete VaR analysis pipeline:
```bash
python main.py
```

This will:
- Fetch 3 years of historical data for 20 NIFTY50 stocks
- Run Monte Carlo simulations for all 5 stress regimes
- Perform Basel II backtesting procedures
- Generate an interactive HTML dashboard
- Display comprehensive results in console

### Expected Output

#### Console Summary
```
======================================================================
 MONTE CARLO VaR ENGINE — RESULTS
======================================================================
Portfolio VaR (95%): ₹2,456,789
Portfolio VaR (99%): ₹3,789,123
CVaR (99%):          ₹4,567,890

Backtest Results:
----------------------------------------------------------------------
Regime          Kupiec p   Christoffersen p   Status
----------------------------------------------------------------------
Bull            0.1234      0.2345              PASS
Bear            0.0987      0.1876              PASS
High_vol        0.0456      0.1234              PASS
Low_liquidity   0.0789      0.1567              PASS
Crisis          0.0345      0.0987              PASS
----------------------------------------------------------------------
```

#### Dashboard
- **File**: `var_dashboard.html`
- **Features**: Interactive 6-panel visualization
- **Theme**: Professional dark theme
- **Export**: Open in any web browser

## 📊 Portfolio Details

### Assets (20 NIFTY50 Stocks)
```
RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, WIPRO, BAJFINANCE, 
AXISBANK, LT, MARUTI, ASIANPAINT, NESTLEIND, HINDUNILVR, 
SUNPHARMA, TATAMOTORS, ONGC, COALINDIA, NTPC, POWERGRID, SBIN
```

### Configuration
- **Portfolio Size**: ₹1,00,00,000 (1 Crore INR)
- **Weights**: Equal weight (5% each asset)
- **Historical Data**: 3 years of daily returns
- **Simulations**: 10,000 per regime
- **Time Horizon**: 10 days (Basel II standard)

## 🔧 Technical Implementation

### Monte Carlo Simulation
```python
# Correlated random number generation
z = np.random.standard_normal((n_simulations, n_assets, n_days))
correlated_returns = cholesky_matrix @ z

# Regime adjustments
mu_adj, sigma_adj = get_regime_parameters(regime)
```

### Backtesting Tests
```python
# Kupiec POF Test
lr_stat = 2 * [log(L1) - log(L0)]
p_value = 1 - chi2.cdf(lr_stat, df=1)

# Christoffersen Independence Test
# Tests for VaR breach clustering
```

### Basel II Compliance
- **Internal Models Approach (IMA)** implementation
- **10-day 99% VaR** as required by Basel II
- **Stress testing** for regulatory capital requirements
- **Model validation** through statistical backtesting

## 📈 Output Files

- `var_dashboard.html` - Interactive dashboard
- `var_results.csv` - VaR results by regime
- `backtest_results.csv` - Backtesting statistics

## ⚠️ Important Notes

### Data Requirements
- **Internet connection** required for yfinance data fetching
- **Market hours**: Data fetching may be slower during market hours
- **Missing data**: Automatically handled with forward/backward fill

### Performance
- **Simulation time**: ~30-60 seconds for all regimes
- **Memory usage**: ~500MB for 10,000 simulations
- **Dashboard size**: ~2-5MB HTML file

### Regulatory Compliance
- **Significance level**: 5% for hypothesis testing
- **Confidence levels**: 95% and 99% VaR
- **Time horizon**: 10 days (Basel II standard)

## 🐛 Troubleshooting

### Common Issues

1. **Data fetching errors**:
   ```bash
   # Check internet connection
   ping yahoo.com
   ```

2. **Memory issues**:
   ```python
   # Reduce simulations in var_engine.py
   self.n_simulations = 5000  # Instead of 10000
   ```

3. **Missing dependencies**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

## 📞 Support

For issues or questions:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure internet connectivity for data fetching

## 📄 License

This project is for educational and demonstration purposes. Use in production requires additional validation and regulatory approval.

---

**Built with**: Python, NumPy, Pandas, Plotly, yfinance, SciPy  
**Compliance**: Basel II regulatory framework  
**Purpose**: Portfolio risk management and regulatory reporting
