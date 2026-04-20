"""
Monte Carlo VaR Engine - Main Pipeline
=====================================

This module orchestrates the complete VaR calculation and reporting workflow
for Basel II compliance testing.

Pipeline Steps:
1. Initialize VaR Engine and fetch market data
2. Calculate covariance matrix and Cholesky decomposition
3. Run Monte Carlo simulations for all stress regimes
4. Perform backtesting with Kupiec and Christoffersen tests
5. Generate interactive dashboard
6. Display comprehensive results

Usage:
    python main.py
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from var_engine import MonteCarloVAREngine
from backtester import VaRBacktester
from report import EnhancedVaRReportGenerator

def print_header():
    """Print application header"""
    print("="*70)
    print(" MONTE CARLO VaR ENGINE - BASEL II COMPLIANT PORTFOLIO RISK SYSTEM")
    print("="*70)
    print("Portfolio: 20 NIFTY50 Assets | Size: ₹1,00,00,000 (1 Cr)")
    print("Simulations: 10,000 per regime | Confidence: 95% & 99%")
    print("="*70)

def print_results_summary(var_results, backtest_results):
    """
    Print formatted results summary to console
    
    Args:
        var_results: Dictionary of VaR results
        backtest_results: DataFrame of backtest results
    """
    print("\n" + "="*70)
    print(" MONTE CARLO VaR ENGINE — RESULTS")
    print("="*70)
    
    # Normal regime results
    normal_results = var_results['normal']
    print(f"Portfolio VaR (95%): ₹{normal_results['var_95']:,.0f}")
    print(f"Portfolio VaR (99%): ₹{normal_results['var_99']:,.0f}")
    print(f"CVaR (99%):          ₹{normal_results['cvar_99']:,.0f}")
    
    # Backtest results
    print("\nBacktest Results:")
    print("-" * 70)
    print(f"{'Regime':<15} {'Kupiec p':<10} {'Christoffersen p':<15} {'Status':<8}")
    print("-" * 70)
    
    for _, row in backtest_results.iterrows():
        # Use average p-values for display
        kupiec_p = (row['Kupiec 95% p-value'] + row['Kupiec 99% p-value']) / 2
        christoffersen_p = (row['Christoffersen 95% p-value'] + row['Christoffersen 99% p-value']) / 2
        
        status_color = "✓" if row['Status'] == 'PASS' else "✗"
        print(f"{row['Regime']:<15} {kupiec_p:<10.4f} {christoffersen_p:<15.4f} {row['Status']:<8} {status_color}")
    
    print("-" * 70)
    
    # Basel II compliance summary
    passing_regimes = sum(backtest_results['Status'] == 'PASS')
    total_regimes = len(backtest_results)
    compliance_status = "COMPLIANT" if passing_regimes == total_regimes else "NON-COMPLIANT"
    
    print(f"\nBasel II Compliance: {compliance_status}")
    print(f"Regimes Passing: {passing_regimes}/{total_regimes}")
    print("="*70)

def print_regime_comparison(var_results):
    """
    Print detailed VaR comparison across all regimes
    
    Args:
        var_results: Dictionary of VaR results for all regimes
    """
    print("\n" + "="*70)
    print(" VaR COMPARISON ACROSS STRESS REGIMES")
    print("="*70)
    print(f"{'Regime':<15} {'VaR 95% (₹)':<15} {'VaR 99% (₹)':<15} {'CVaR 99% (₹)':<15}")
    print("-" * 70)
    
    for regime, results in var_results.items():
        print(f"{regime.upper():<15} {results['var_95']:<15,.0f} {results['var_99']:<15,.0f} {results['cvar_99']:<15,.0f}")
    
    print("="*70)

def main():
    """Main execution pipeline"""
    try:
        # Print header
        print_header()
        
        # Step 1: Initialize VaR Engine
        print("\n[STEP 1] Initializing Monte Carlo VaR Engine...")
        var_engine = MonteCarloVAREngine(portfolio_size=100000000)  # 1 Cr INR
        
        # Step 2: Fetch market data
        print("\n[STEP 2] Fetching historical market data...")
        var_engine.fetch_data(years=3)
        
        # Step 3: Calculate covariance and Cholesky decomposition
        print("\n[STEP 3] Calculating covariance matrix and Cholesky decomposition...")
        var_engine.calculate_cholesky()
        
        # Step 4: Run Monte Carlo simulations for all regimes
        print("\n[STEP 4] Running Monte Carlo simulations...")
        var_results = var_engine.run_all_regimes()
        
        # Step 5: Initialize backtester and run backtests
        print("\n[STEP 5] Running Basel II backtesting procedures...")
        backtester = VaRBacktester(significance_level=0.05)
        
        # Generate historical returns for backtesting
        historical_returns = var_engine.returns
        
        # Run backtests for all regimes
        backtest_results = backtester.backtest_all_regimes(var_results, historical_returns)
        
        # Step 6: Generate interactive dashboard
        print("\n[STEP 6] Generating interactive dashboard...")
        report_generator = EnhancedVaRReportGenerator(var_engine, backtester)
        dashboard_file = report_generator.generate_dashboard(var_results, backtest_results)
        
        # Step 7: Display results
        print_results_summary(var_results, backtest_results)
        print_regime_comparison(var_results)
        
        # Step 8: Generate backtest report
        backtest_report = backtester.generate_backtest_report(backtest_results)
        print(backtest_report)
        
        # Final status
        print(f"\n✓ Enhanced dashboard saved: {dashboard_file}")
        print("✓ Monte Carlo VaR Engine execution completed successfully!")
        
        # Export results to CSV for further analysis
        export_results_to_csv(var_results, backtest_results)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Please check the error message above and try again.")
        sys.exit(1)

def export_results_to_csv(var_results, backtest_results):
    """
    Export results to CSV files for further analysis
    
    Args:
        var_results: Dictionary of VaR results
        backtest_results: DataFrame of backtest results
    """
    try:
        # Export VaR results
        var_data = []
        for regime, results in var_results.items():
            var_data.append({
                'Regime': regime,
                'VaR_95': results['var_95'],
                'VaR_99': results['var_99'],
                'CVaR_95': results['cvar_95'],
                'CVaR_99': results['cvar_99']
            })
        
        var_df = pd.DataFrame(var_data)
        var_df.to_csv('var_results.csv', index=False)
        
        # Export backtest results
        backtest_results.to_csv('backtest_results.csv', index=False)
        
        print(f"\n✓ Results exported:")
        print(f"  - VaR results: var_results.csv")
        print(f"  - Backtest results: backtest_results.csv")
        
    except Exception as e:
        print(f"⚠ Warning: Could not export results to CSV: {str(e)}")

if __name__ == "__main__":
    main()
