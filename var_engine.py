"""
Monte Carlo Value-at-Risk Engine - Basel II Compliant
===============================================

This module implements a production-grade Monte Carlo VaR calculation system
with stress regime modeling for Basel II compliance requirements.

Key Features:
- 20-asset NIFTY50 portfolio simulation
- Full covariance matrix with Cholesky decomposition
- 5 stress regimes for comprehensive risk assessment
- VaR and CVaR (Expected Shortfall) calculations at 95% and 99% confidence levels
- 10,000 Monte Carlo simulations per scenario
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class MonteCarloVAREngine:
    """
    Monte Carlo VaR Engine for portfolio risk assessment
    
    Basel II Relevance:
    - Implements Internal Models Approach (IMA) for market risk
    - Provides 10-day 99% VaR as required by Basel II
    - Includes stress testing for regulatory capital requirements
    """
    
    def __init__(self, portfolio_size: float = 100000000):
        """
        Initialize VaR Engine
        
        Args:
            portfolio_size: Portfolio value in INR (default: 1 Cr = ₹1,00,00,000)
        """
        self.portfolio_size = portfolio_size
        self.n_assets = 20
        self.n_simulations = 10000
        
        # NIFTY50 tickers - realistic selection
        self.tickers = [
            'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
            'WIPRO.NS', 'BAJFINANCE.NS', 'AXISBANK.NS', 'LT.NS', 'MARUTI.NS',
            'ASIANPAINT.NS', 'NESTLEIND.NS', 'HINDUNILVR.NS', 'SUNPHARMA.NS',
            'TATAMOTORS.NS', 'ONGC.NS', 'COALINDIA.NS', 'NTPC.NS', 'POWERGRID.NS', 'SBIN.NS'
        ]
        
        # Equal weight portfolio (5% each)
        self.weights = np.array([0.05] * self.n_assets)
        
        # Data storage
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.cholesky_matrix = None
        
    def fetch_data(self, years: int = 3) -> None:
        """
        Fetch historical data for NIFTY50 assets
        
        Args:
            years: Number of years of historical data to fetch
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            print(f"Fetching {years} years of historical data...")
            data = yf.download(self.tickers, start=start_date, end=end_date)['Adj Close']
            
            # Handle missing data - forward fill then backward fill
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate daily returns
            self.returns = data.pct_change().dropna()
            
            # Check if we got valid data
            if self.returns.shape[0] == 0:
                raise ValueError("No valid data fetched")
                
        except Exception as e:
            print(f"Warning: Could not fetch live data ({str(e)})")
            print("Using simulated data for demonstration...")
            self.returns = self._generate_simulated_data()
            
        # Calculate mean returns and covariance matrix
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        print(f"Data prepared successfully. Shape: {self.returns.shape}")
        
    def _generate_simulated_data(self, years: int = 3) -> pd.DataFrame:
        """
        Generate simulated historical data for demonstration
        
        Args:
            years: Number of years of simulated data
            
        Returns:
            DataFrame of simulated returns
        """
        np.random.seed(42)  # For reproducible results
        
        # Generate trading days (approximately 252 per year)
        n_days = years * 252
        
        # Create realistic correlation matrix
        # Indian stocks tend to have moderate to high correlation
        base_correlation = 0.3
        correlation_matrix = np.full((self.n_assets, self.n_assets), base_correlation)
        np.fill_diagonal(correlation_matrix, 1.0)
        
        # Add some sector-specific correlations
        # IT stocks: TCS, INFY, WIPRO
        it_indices = [1, 2, 5]  # TCS, INFY, WIPRO
        for i in it_indices:
            for j in it_indices:
                correlation_matrix[i, j] = 0.6
                
        # Banking stocks: HDFCBANK, ICICIBANK, AXISBANK, SBIN
        bank_indices = [3, 4, 7, 19]  # HDFCBANK, ICICIBANK, AXISBANK, SBIN
        for i in bank_indices:
            for j in bank_indices:
                correlation_matrix[i, j] = 0.7
        
        # Generate correlated returns using Cholesky
        try:
            chol = np.linalg.cholesky(correlation_matrix)
        except:
            # If Cholesky fails, use nearest positive definite matrix
            correlation_matrix = self._nearest_positive_definite(correlation_matrix)
            chol = np.linalg.cholesky(correlation_matrix)
        
        # Generate uncorrelated random returns
        uncorrelated_returns = np.random.standard_normal((n_days, self.n_assets))
        
        # Apply correlation structure
        correlated_returns = uncorrelated_returns @ chol.T
        
        # Add realistic return characteristics
        # Annual returns: mean ~12%, volatility ~18% for Indian market
        annual_mean = 0.12
        annual_vol = 0.18
        daily_mean = annual_mean / 252
        daily_vol = annual_vol / np.sqrt(252)
        
        # Scale and add drift
        returns = correlated_returns * daily_vol + daily_mean
        
        # Add some fat tails (occasional large moves)
        fat_tail_events = np.random.choice(n_days, size=int(n_days * 0.02), replace=False)
        for event_day in fat_tail_events:
            multiplier = np.random.uniform(2, 4)
            returns[event_day, :] *= multiplier
        
        # Create DataFrame
        return pd.DataFrame(
            returns,
            index=pd.date_range(end=datetime.now(), periods=n_days, freq='B'),
            columns=self.tickers
        )
        
    def _nearest_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """
        Find nearest positive definite matrix to input matrix
        
        Args:
            matrix: Input matrix
            
        Returns:
            Nearest positive definite matrix
        """
        B = (matrix + matrix.T) / 2
        _, s, V = np.linalg.svd(B)
        
        H = V.T @ np.diag(s) @ V
        
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        if self._is_positive_definite(A3):
            return A3
            
        spacing = np.spacing(np.linalg.norm(matrix))
        I = np.eye(matrix.shape[0])
        k = 1
        while not self._is_positive_definite(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
            
        return A3
        
    def _is_positive_definite(self, matrix: np.ndarray) -> bool:
        """
        Check if matrix is positive definite
        
        Args:
            matrix: Input matrix
            
        Returns:
            True if positive definite, False otherwise
        """
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False
        
    def calculate_cholesky(self) -> None:
        """
        Calculate Cholesky decomposition for correlated random variables
        
        Basel II Relevance:
        - Essential for generating correlated asset price paths
        - Ensures realistic dependency structure in simulations
        """
        try:
            # Ensure positive definite covariance matrix
            eigenvals = np.linalg.eigvals(self.cov_matrix)
            if np.any(eigenvals < 0):
                # Add small positive constant to diagonal
                self.cov_matrix += np.eye(self.n_assets) * 1e-8
                
            self.cholesky_matrix = np.linalg.cholesky(self.cov_matrix)
            print("Cholesky decomposition calculated successfully")
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Covariance matrix not positive definite: {e}")
            
    def generate_correlated_returns(self, n_days: int = 1, regime: str = 'normal') -> np.ndarray:
        """
        Generate correlated returns using Cholesky decomposition
        
        Args:
            n_days: Number of days to simulate
            regime: Market regime for stress testing
            
        Returns:
            Array of simulated returns (n_simulations x n_assets x n_days)
        """
        # Generate uncorrelated random numbers
        z = np.random.standard_normal((self.n_simulations, self.n_assets, n_days))
        
        # Apply regime adjustments
        mu_adj, sigma_adj = self._get_regime_parameters(regime)
        
        # Generate correlated returns
        correlated_returns = np.zeros_like(z)
        for i in range(self.n_simulations):
            for day in range(n_days):
                # Apply Cholesky transformation
                correlated_returns[i, :, day] = mu_adj + np.dot(self.cholesky_matrix, z[i, :, day]) * sigma_adj
                
        return correlated_returns
        
    def _get_regime_parameters(self, regime: str) -> Tuple[np.ndarray, float]:
        """
        Get regime-specific parameters for stress testing
        
        Args:
            regime: Market regime name
            
        Returns:
            Tuple of (mean_adjustment, volatility_scaling)
        """
        regimes = {
            'normal': (self.mean_returns.values, 1.0),
            'bull': (self.mean_returns.values * 1.5, 0.7),  # Positive drift, low vol
            'bear': (self.mean_returns.values * -0.5, 1.2),  # Negative drift, moderate vol
            'high_vol': (self.mean_returns.values, 2.0),  # 2x volatility
            'low_liquidity': (self.mean_returns.values, 1.5),  # Fat tails handled in simulation
            'crisis': (self.mean_returns.values - 0.03, 1.8)  # -3% shock to all assets
        }
        
        return regimes.get(regime, regimes['normal'])
        
    def simulate_portfolio(self, regime: str = 'normal', n_days: int = 10) -> Dict:
        """
        Run Monte Carlo simulation for portfolio
        
        Args:
            regime: Market regime
            n_days: Number of days to simulate (10 for Basel II)
            
        Returns:
            Dictionary with VaR, CVaR, and P&L distribution
        """
        if regime == 'low_liquidity':
            # Use t-distribution for fat tails
            returns = self._simulate_t_distribution(n_days)
        else:
            returns = self.generate_correlated_returns(n_days, regime)
            
        # Calculate portfolio returns for each simulation
        portfolio_returns = np.zeros(self.n_simulations)
        for i in range(self.n_simulations):
            # Calculate portfolio return for each day, then compound
            daily_portfolio_returns = np.dot(returns[i, :, :].T, self.weights)
            cumulative_return = np.prod(1 + daily_portfolio_returns) - 1
            portfolio_returns[i] = cumulative_return
            
        # Calculate P&L in INR
        portfolio_pnl = portfolio_returns * self.portfolio_size
        
        # Calculate VaR and CVaR
        var_95 = np.percentile(portfolio_pnl, 5)
        var_99 = np.percentile(portfolio_pnl, 1)
        
        # CVaR (Expected Shortfall) - average loss beyond VaR
        cvar_95 = portfolio_pnl[portfolio_pnl <= var_95].mean()
        cvar_99 = portfolio_pnl[portfolio_pnl <= var_99].mean()
        
        return {
            'var_95': abs(var_95),
            'var_99': abs(var_99),
            'cvar_95': abs(cvar_95),
            'cvar_99': abs(cvar_99),
            'pnl_distribution': portfolio_pnl,
            'regime': regime
        }
        
    def _simulate_t_distribution(self, n_days: int) -> np.ndarray:
        """
        Simulate returns using t-distribution for fat tails (low liquidity regime)
        
        Args:
            n_days: Number of days to simulate
            
        Returns:
            Array of simulated returns
        """
        df = 5  # Degrees of freedom for fat tails
        t_returns = np.random.standard_t(df, (self.n_simulations, self.n_assets, n_days))
        
        # Scale to match covariance matrix
        t_returns = t_returns * np.sqrt((df - 2) / df)  # Adjust for t-distribution variance
        
        # Apply Cholesky transformation
        correlated_returns = np.zeros_like(t_returns)
        for i in range(self.n_simulations):
            for day in range(n_days):
                correlated_returns[i, :, day] = self.mean_returns.values + np.dot(self.cholesky_matrix, t_returns[i, :, day])
                
        return correlated_returns
        
    def run_all_regimes(self) -> Dict[str, Dict]:
        """
        Run simulations for all stress regimes
        
        Returns:
            Dictionary of results for each regime
        """
        regimes = ['normal', 'bull', 'bear', 'high_vol', 'low_liquidity', 'crisis']
        results = {}
        
        print("Running Monte Carlo simulations for all regimes...")
        for regime in regimes:
            print(f"Simulating {regime} regime...")
            results[regime] = self.simulate_portfolio(regime)
            
        return results
        
    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Get correlation matrix of assets
        
        Returns:
            Correlation matrix DataFrame
        """
        if self.returns is None:
            raise ValueError("Data not fetched. Call fetch_data() first.")
            
        return self.returns.corr()
        
    def get_cholesky_factors(self) -> pd.DataFrame:
        """
        Get Cholesky decomposition matrix for visualization
        
        Returns:
            Lower triangular Cholesky matrix as DataFrame
        """
        if self.cholesky_matrix is None:
            raise ValueError("Cholesky decomposition not calculated. Call calculate_cholesky() first.")
            
        return pd.DataFrame(
            self.cholesky_matrix,
            index=self.tickers,
            columns=[f'Factor_{i+1}' for i in range(self.n_assets)]
        )
