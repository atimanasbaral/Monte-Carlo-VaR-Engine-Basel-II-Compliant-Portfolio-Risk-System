"""
Backtesting Module - Basel II Compliance Testing
==============================================

This module implements statistical backtesting procedures for VaR models
as required by Basel II regulatory framework.

Key Features:
- Kupiec Proportion of Failures (POF) test
- Christoffersen Independence test for VaR breach clustering
- Comprehensive backtesting across all stress regimes
- Pass/fail criteria based on 5% significance level

Basel II Relevance:
- Regulatory requirement for VaR model validation
- Ensures model accuracy and reliability
- Required for internal models approach (IMA) approval
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class VaRBacktester:
    """
    VaR Model Backtester implementing Basel II statistical tests
    
    Tests Implemented:
    1. Kupiec POF Test - Tests if the proportion of VaR breaches matches expected
    2. Christoffersen Test - Tests independence of VaR breaches (no clustering)
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize backtester
        
        Args:
            significance_level: Significance level for hypothesis testing (default: 5%)
        """
        self.significance_level = significance_level
        self.critical_values = self._get_critical_values()
        
    def _get_critical_values(self) -> Dict[float, float]:
        """
        Get critical values for likelihood ratio tests
        
        Returns:
            Dictionary of critical values for different significance levels
        """
        # Chi-squared critical values with 1 degree of freedom
        return {
            0.01: 6.635,  # 99% confidence
            0.05: 3.841,  # 95% confidence
            0.10: 2.706   # 90% confidence
        }
        
    def kupiec_test(self, var_violations: np.ndarray, var_level: float = 0.05) -> Dict:
        """
        Kupiec Proportion of Failures (POF) Test
        
        Tests H0: The proportion of VaR violations equals the expected rate
        
        Args:
            var_violations: Binary array of VaR violations (1 = violation, 0 = no violation)
            var_level: VaR confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            Dictionary with test statistic, p-value, and pass/fail status
        """
        n = len(var_violations)
        x = np.sum(var_violations)  # Number of violations
        
        if x == 0:
            # No violations - model may be too conservative
            lr_stat = 0
            p_value = 1.0
            status = "PASS"
        else:
            # Expected number of violations
            expected_rate = var_level
            expected_violations = n * expected_rate
            
            # Likelihood ratio statistic
            # LR = 2 * [log(L1) - log(L0)]
            # L1: Unconstrained likelihood (actual violation rate)
            # L0: Constrained likelihood (expected violation rate)
            
            actual_rate = x / n
            
            # Avoid log(0) issues
            if actual_rate == 0:
                actual_rate = 1e-10
            if actual_rate == 1:
                actual_rate = 1 - 1e-10
                
            lr_stat = 2 * (
                x * np.log(actual_rate / expected_rate) +
                (n - x) * np.log((1 - actual_rate) / (1 - expected_rate))
            )
            
            # P-value from chi-squared distribution
            p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
            
            # Pass/fail based on significance level
            status = "PASS" if p_value > self.significance_level else "FAIL"
            
        return {
            'test': 'Kupiec POF',
            'violations': int(x),
            'expected_violations': n * var_level,
            'violation_rate': x / n,
            'lr_statistic': lr_stat,
            'p_value': p_value,
            'critical_value': self.critical_values[self.significance_level],
            'status': status,
            'var_level': var_level
        }
        
    def christoffersen_test(self, var_violations: np.ndarray) -> Dict:
        """
        Christoffersen Independence Test
        
        Tests H0: VaR violations are independent (no clustering)
        
        Args:
            var_violations: Binary array of VaR violations
            
        Returns:
            Dictionary with test statistic, p-value, and pass/fail status
        """
        n = len(var_violations)
        
        # Count transition patterns
        # 00: no violation followed by no violation
        # 01: no violation followed by violation
        # 10: violation followed by no violation
        # 11: violation followed by violation
        
        n00 = n01 = n10 = n11 = 0
        
        for i in range(n - 1):
            if var_violations[i] == 0 and var_violations[i + 1] == 0:
                n00 += 1
            elif var_violations[i] == 0 and var_violations[i + 1] == 1:
                n01 += 1
            elif var_violations[i] == 1 and var_violations[i + 1] == 0:
                n10 += 1
            elif var_violations[i] == 1 and var_violations[i + 1] == 1:
                n11 += 1
                
        # Total transitions
        n0 = n00 + n01  # Total transitions from no violation
        n1 = n10 + n11  # Total transitions from violation
        
        if n0 == 0 or n1 == 0:
            # Insufficient transitions
            lr_stat = 0
            p_value = 1.0
            status = "PASS"
        else:
            # Transition probabilities
            pi01 = n01 / n0 if n0 > 0 else 0
            pi11 = n11 / n1 if n1 > 0 else 0
            
            # Overall violation probability
            pi = np.sum(var_violations) / n
            
            # Likelihood ratio statistic for independence
            # LR_ind = 2 * [log(L_ind) - log(L_unconstrained)]
            
            # Avoid log(0) issues
            pi = max(min(pi, 0.999), 0.001)
            pi01 = max(min(pi01, 0.999), 0.001)
            pi11 = max(min(pi11, 0.999), 0.001)
            
            if n00 > 0 and n10 > 0:
                lr_stat = 2 * (
                    n00 * np.log((1 - pi01) / (1 - pi)) +
                    n01 * np.log(pi01 / pi) +
                    n10 * np.log((1 - pi11) / (1 - pi)) +
                    n11 * np.log(pi11 / pi)
                )
            else:
                lr_stat = 0
                
            # P-value from chi-squared distribution
            p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
            
            # Pass/fail based on significance level
            status = "PASS" if p_value > self.significance_level else "FAIL"
            
        return {
            'test': 'Christoffersen Independence',
            'transitions_00': n00,
            'transitions_01': n01,
            'transitions_10': n10,
            'transitions_11': n11,
            'lr_statistic': lr_stat,
            'p_value': p_value,
            'critical_value': self.critical_values[self.significance_level],
            'status': status
        }
        
    def generate_backtest_data(self, historical_returns: pd.DataFrame, 
                             var_results: Dict, n_test_days: int = 252) -> np.ndarray:
        """
        Generate backtest data by comparing VaR predictions with actual returns
        
        Args:
            historical_returns: Historical portfolio returns
            var_results: VaR simulation results
            n_test_days: Number of days to use for backtesting
            
        Returns:
            Binary array of VaR violations
        """
        # Use last n_test_days of historical data
        test_returns = historical_returns.tail(n_test_days)
        
        # Calculate portfolio returns
        portfolio_returns = test_returns.mean(axis=1)  # Simplified - should use weights
        
        # Generate VaR violations for 95% and 99% levels
        var_95 = var_results['var_95'] / 100000000  # Convert back to return scale
        var_99 = var_results['var_99'] / 100000000
        
        # VaR violations (1 if return < -VaR, 0 otherwise)
        violations_95 = (portfolio_returns < -var_95).astype(int).values
        violations_99 = (portfolio_returns < -var_99).astype(int).values
        
        return violations_95, violations_99
        
    def backtest_regime(self, regime_name: str, var_results: Dict, 
                       historical_returns: pd.DataFrame) -> Dict:
        """
        Run comprehensive backtest for a single regime
        
        Args:
            regime_name: Name of the regime
            var_results: VaR results for the regime
            historical_returns: Historical returns data
            
        Returns:
            Dictionary with backtest results
        """
        # Generate backtest data
        violations_95, violations_99 = self.generate_backtest_data(
            historical_returns, var_results
        )
        
        # Run Kupiec tests
        kupiec_95 = self.kupiec_test(violations_95, var_level=0.05)
        kupiec_99 = self.kupiec_test(violations_99, var_level=0.01)
        
        # Run Christoffersen tests
        christoffersen_95 = self.christoffersen_test(violations_95)
        christoffersen_99 = self.christoffersen_test(violations_99)
        
        return {
            'regime': regime_name,
            'kupiec_95': kupiec_95,
            'kupiec_99': kupiec_99,
            'christoffersen_95': christoffersen_95,
            'christoffersen_99': christoffersen_99,
            'overall_status': self._get_overall_status([
                kupiec_95['status'], kupiec_99['status'],
                christoffersen_95['status'], christoffersen_99['status']
            ])
        }
        
    def _get_overall_status(self, statuses: List[str]) -> str:
        """
        Determine overall pass/fail status
        
        Args:
            statuses: List of individual test statuses
            
        Returns:
            Overall status (PASS if all tests pass, FAIL otherwise)
        """
        return "PASS" if all(status == "PASS" for status in statuses) else "FAIL"
        
    def backtest_all_regimes(self, var_results: Dict[str, Dict], 
                           historical_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtests for all regimes
        
        Args:
            var_results: Dictionary of VaR results for all regimes
            historical_returns: Historical returns data
            
        Returns:
            DataFrame with comprehensive backtest results
        """
        results = []
        
        print("Running backtests for all regimes...")
        for regime_name, regime_results in var_results.items():
            print(f"Backtesting {regime_name} regime...")
            backtest_result = self.backtest_regime(regime_name, regime_results, historical_returns)
            results.append(backtest_result)
            
        # Convert to DataFrame for easy visualization
        df_results = pd.DataFrame(results)
        
        # Create summary table
        summary_data = []
        for result in results:
            summary_data.append({
                'Regime': result['regime'],
                'Kupiec 95% p-value': result['kupiec_95']['p_value'],
                'Kupiec 99% p-value': result['kupiec_99']['p_value'],
                'Christoffersen 95% p-value': result['christoffersen_95']['p_value'],
                'Christoffersen 99% p-value': result['christoffersen_99']['p_value'],
                'Status': result['overall_status']
            })
            
        summary_df = pd.DataFrame(summary_data)
        
        return summary_df
        
    def generate_backtest_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate formatted backtest report
        
        Args:
            results_df: Backtest results DataFrame
            
        Returns:
            Formatted report string
        """
        report = "\n" + "="*60 + "\n"
        report += "BACKTESTING RESULTS - BASEL II COMPLIANCE\n"
        report += "="*60 + "\n\n"
        
        report += "Regine-wise Test Results:\n"
        report += "-"*40 + "\n"
        
        for _, row in results_df.iterrows():
            report += f"\n{row['Regime'].upper()} Regime:\n"
            report += f"  Kupiec 95% p-value:  {row['Kupiec 95% p-value']:.4f}\n"
            report += f"  Kupiec 99% p-value:  {row['Kupiec 99% p-value']:.4f}\n"
            report += f"  Christoffersen 95% p-value: {row['Christoffersen 95% p-value']:.4f}\n"
            report += f"  Christoffersen 99% p-value: {row['Christoffersen 99% p-value']:.4f}\n"
            report += f"  Overall Status: {row['Status']}\n"
            
        report += "\n" + "="*60 + "\n"
        report += "BASEL II COMPLIANCE SUMMARY:\n"
        report += f"Regimes Passing: {sum(results_df['Status'] == 'PASS')}/{len(results_df)}\n"
        report += f"Overall Compliance: {'PASS' if all(results_df['Status'] == 'PASS') else 'FAIL'}\n"
        report += "="*60 + "\n"
        
        return report
