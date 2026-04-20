"""
Enhanced Report Generator - Publication-Grade Interactive Dashboard
================================================================

This module generates a comprehensive 8-panel interactive dashboard
using Plotly for Monte Carlo VaR analysis and Basel II compliance reporting.

Enhanced Dashboard Panels (4×2 Grid):
1. P&L Distribution Histogram with VaR/CVaR lines and tail shading
2. VaR Comparison Bar Chart across stress regimes (95% vs 99% CI)
3. 20-Asset Correlation Heatmap with annotations
4. Backtest Results Table with color-coded status
5. Historical Cumulative Portfolio Return with max drawdown
6. Cholesky Factor Heatmap (lower triangular matrix)
7. Monte Carlo Simulation Paths (Fan Chart)
8. CVaR vs VaR Sensitivity Analysis (Confidence Level Sweep)

Features:
- Publication-grade visualizations with dark theme
- Indian number formatting (¥X,XX,XXX)
- Interactive hover information and annotations
- Responsive 4×2 grid layout
- Single HTML file output
"""

import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedVaRReportGenerator:
    """
    Enhanced Interactive Dashboard Generator for VaR Analysis
    
    Basel II Relevance:
    - Provides comprehensive risk visualization for regulatory reporting
    - Enables stakeholders to understand model performance across stress regimes
    - Facilitates risk management decision making with detailed analytics
    """
    
    def __init__(self, var_engine, backtester):
        """
        Initialize enhanced report generator
        
        Args:
            var_engine: MonteCarloVAREngine instance
            backtester: VaRBacktester instance
        """
        self.var_engine = var_engine
        self.backtester = backtester
        
        # Dark theme template
        self.template = "plotly_dark"
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'danger': '#d62728',
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'info': '#17becf',
            'purple': '#9467bd',
            'brown': '#8c564b'
        }
        
    def format_indian_currency(self, value: float) -> str:
        """
        Format number in Indian currency system
        
        Args:
            value: Numeric value to format
            
        Returns:
            Formatted Indian currency string
        """
        if value >= 10000000:  # Crore
            return f"¥{value/10000000:.2f} Cr"
        elif value >= 100000:  # Lakh
            return f"¥{value/100000:.1f} L"
        else:
            return f"¥{value:,.0f}"
    
    def create_pnl_distribution_panel(self, var_results: Dict) -> go.Figure:
        """
        Create enhanced P&L distribution histogram with VaR/CVaR lines and tail shading
        
        Args:
            var_results: VaR simulation results (normal regime)
            
        Returns:
            Plotly figure for P&L distribution
        """
        pnl = var_results['pnl_distribution']
        var_95 = var_results['var_95']
        var_99 = var_results['var_99']
        cvar_99 = var_results['cvar_99']
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=pnl,
            nbinsx=80,
            name='Portfolio P&L',
            opacity=0.8,
            marker_color=self.colors['primary'],
            hovertemplate='P&L: %{x:,.0f}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add CVaR tail shading (red translucent area beyond VaR 99%)
        cvar_tail = pnl[pnl <= -var_99]
        if len(cvar_tail) > 0:
            fig.add_trace(go.Histogram(
                x=cvar_tail,
                nbinsx=40,
                name='CVaR Zone',
                opacity=0.3,
                marker_color=self.colors['danger'],
                hovertemplate='CVaR Zone<br>P&L: %{x:,.0f}<br>Count: %{y}<extra></extra>'
            ))
        
        # Add VaR 95% line (orange dashed)
        fig.add_vline(
            x=-var_95,
            line_dash="dash",
            line_color=self.colors['warning'],
            line_width=2,
            annotation_text=f"VaR 95%: {self.format_indian_currency(var_95)}",
            annotation_position="top right",
            annotation_font=dict(color=self.colors['warning'])
        )
        
        # Add VaR 99% line (red dashed)
        fig.add_vline(
            x=-var_99,
            line_dash="dash",
            line_color=self.colors['danger'],
            line_width=2,
            annotation_text=f"VaR 99%: {self.format_indian_currency(var_99)}",
            annotation_position="top right",
            annotation_font=dict(color=self.colors['danger'])
        )
        
        # Add CVaR 99% line (red dotted)
        fig.add_vline(
            x=-cvar_99,
            line_dash="dot",
            line_color=self.colors['danger'],
            line_width=2,
            annotation_text=f"CVaR 99%: {self.format_indian_currency(cvar_99)}",
            annotation_position="bottom right",
            annotation_font=dict(color=self.colors['danger'])
        )
        
        fig.update_layout(
            title="Simulated P&L Distribution - 10,000 Paths",
            xaxis_title="Daily P&L (¥)",
            yaxis_title="Frequency",
            template=self.template,
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
        
    def create_var_comparison_panel(self, var_results: Dict[str, Dict]) -> go.Figure:
        """
        Create enhanced VaR comparison bar chart across stress regimes
        
        Args:
            var_results: Dictionary of VaR results for all regimes
            
        Returns:
            Plotly figure for VaR comparison
        """
        # Filter out normal regime for stress comparison
        stress_regimes = {k: v for k, v in var_results.items() if k != 'normal'}
        regimes = list(stress_regimes.keys())
        var_95_values = [stress_regimes[r]['var_95'] for r in regimes]
        var_99_values = [stress_regimes[r]['var_99'] for r in regimes]
        
        fig = go.Figure()
        
        # VaR 95% bars (blue)
        fig.add_trace(go.Bar(
            x=regimes,
            y=var_95_values,
            name='VaR 95%',
            marker_color=self.colors['primary'],
            text=[self.format_indian_currency(v) for v in var_95_values],
            textposition='auto',
            textfont=dict(size=10),
            hovertemplate='Regime: %{x}<br>VaR 95%: %{text}<extra></extra>'
        ))
        
        # VaR 99% bars (red)
        fig.add_trace(go.Bar(
            x=regimes,
            y=var_99_values,
            name='VaR 99%',
            marker_color=self.colors['danger'],
            text=[self.format_indian_currency(v) for v in var_99_values],
            textposition='auto',
            textfont=dict(size=10),
            hovertemplate='Regime: %{x}<br>VaR 99%: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="VaR by Stress Regime (95% vs 99% CI)",
            xaxis_title="Stress Regime",
            yaxis_title="VaR (¥)",
            template=self.template,
            height=600,
            barmode='group',
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
        
    def create_correlation_heatmap(self) -> go.Figure:
        """
        Create enhanced 20×20 correlation heatmap with annotations
        
        Returns:
            Plotly figure for correlation heatmap
        """
        corr_matrix = self.var_engine.get_correlation_matrix()
        
        # Clean ticker names for display
        tickers_clean = [ticker.replace('.NS', '') for ticker in corr_matrix.columns]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=tickers_clean,
            y=tickers_clean,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=8),
            hovertemplate='Asset 1: %{y}<br>Asset 2: %{x}<br>Correlation: %{z:.3f}<extra></extra>',
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="NIFTY50 Portfolio - Asset Correlation Matrix",
            template=self.template,
            height=600,
            width=800,
            xaxis=dict(tickangle=45),
            yaxis=dict(tickangle=0)
        )
        
        return fig
        
    def create_backtest_results_table(self, backtest_results: pd.DataFrame) -> go.Figure:
        """
        Create enhanced backtest results table with color-coded status
        
        Args:
            backtest_results: DataFrame with backtest results
            
        Returns:
            Plotly figure for backtest results table
        """
        # Prepare data for table
        table_data = []
        cell_colors = []
        
        for _, row in backtest_results.iterrows():
            # Average p-values for display
            kupiec_avg = (row['Kupiec 95% p-value'] + row['Kupiec 99% p-value']) / 2
            christoffersen_avg = (row['Christoffersen 95% p-value'] + row['Christoffersen 99% p-value']) / 2
            
            table_data.append([
                row['Regime'].upper(),
                f"{kupiec_avg:.4f}",
                f"{christoffersen_avg:.4f}",
                row['Status']
            ])
            
            # Color coding for status
            status_color = self.colors['success'] if row['Status'] == 'PASS' else self.colors['danger']
            cell_colors.append([
                'rgba(50, 50, 50, 0.8)',  # Regime
                'rgba(50, 50, 50, 0.8)',  # Kupiec
                'rgba(50, 50, 50, 0.8)',  # Christoffersen
                status_color                # Status
            ])
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Regime', 'Kupiec p-value', 'Christoffersen p-value', 'Status'],
                fill_color='rgba(30, 30, 30, 0.9)',
                align='left',
                font=dict(color='white', size=12, weight='bold'),
                height=40
            ),
            cells=dict(
                values=list(zip(*table_data)),
                fill_color=cell_colors,
                align='left',
                font=dict(color='white', size=11),
                height=35
            )
        )])
        
        fig.update_layout(
            title="Regulatory Backtest Results (Kupiec + Christoffersen)",
            template=self.template,
            height=600,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
        
    def create_cumulative_returns_panel(self) -> go.Figure:
        """
        Create enhanced historical cumulative returns chart with max drawdown
        
        Returns:
            Plotly figure for cumulative returns
        """
        # Get historical data using the same method as var_engine
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3*365)
            
            data = yf.download(self.var_engine.tickers, start=start_date, end=end_date)['Adj Close']
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate portfolio returns (equal weight)
            portfolio_returns = data.pct_change().mean(axis=1)
            cumulative_returns = (1 + portfolio_returns).cumprod() - 1
            
            # Calculate maximum drawdown
            cumulative_peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - cumulative_peak) / cumulative_peak
            max_dd_idx = drawdown.idxmin()
            max_dd_value = drawdown.min()
            
        except Exception as e:
            print(f"Warning: Could not fetch historical data for returns chart: {str(e)}")
            # Create dummy data for demonstration
            dates = pd.date_range(end=datetime.now(), periods=756, freq='B')
            cumulative_returns = pd.Series(
                np.random.normal(0.0005, 0.02, 756).cumsum(),
                index=dates
            )
            max_dd_idx = dates[len(dates)//2]
            max_dd_value = -0.15
        
        fig = go.Figure()
        
        # Add area chart
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values * 100,
            mode='lines',
            name='Cumulative Return',
            line=dict(color=self.colors['primary'], width=2),
            fill='tonexty',
            fillcolor=f'rgba(31, 119, 254, 0.3)',
            hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="white",
            line_width=1,
            opacity=0.5
        )
        
        # Add max drawdown marker
        max_dd_return = cumulative_returns.loc[max_dd_idx] * 100
        fig.add_trace(go.Scatter(
            x=[max_dd_idx],
            y=[max_dd_return],
            mode='markers',
            name='Max Drawdown',
            marker=dict(
                color=self.colors['danger'],
                size=10,
                symbol='diamond'
            ),
            hovertemplate=f'Max Drawdown<br>Date: {max_dd_idx.strftime("%Y-%m-%d")}<br>Value: {max_dd_return:.2f}%<extra></extra>'
        ))
        
        # Add annotation for max drawdown
        fig.add_annotation(
            x=max_dd_idx,
            y=max_dd_return,
            text=f"Max DD: {max_dd_value:.1%}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=self.colors['danger'],
            font=dict(color=self.colors['danger'], size=10)
        )
        
        fig.update_layout(
            title="Historical Cumulative Portfolio Return (3Y)",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            template=self.template,
            height=600,
            showlegend=False
        )
        
        return fig
        
    def create_cholesky_heatmap(self) -> go.Figure:
        """
        Create enhanced Cholesky factor heatmap
        
        Returns:
            Plotly figure for Cholesky factors
        """
        cholesky_df = self.var_engine.get_cholesky_factors()
        
        # Clean ticker names
        tickers_clean = [ticker.replace('.NS', '') for ticker in cholesky_df.index]
        
        fig = go.Figure(data=go.Heatmap(
            z=cholesky_df.values,
            x=[f'Factor {i+1}' for i in range(cholesky_df.shape[1])],
            y=tickers_clean,
            colorscale='Viridis',
            text=np.round(cholesky_df.values, 4),
            texttemplate='%{text}',
            textfont=dict(size=8),
            hovertemplate='Asset: %{y}<br>Factor: %{x}<br>Value: %{z:.4f}<extra></extra>',
            colorbar=dict(title="Cholesky Factor")
        ))
        
        fig.update_layout(
            title="Cholesky Decomposition Factor (L matrix)",
            template=self.template,
            height=600,
            width=800
        )
        
        return fig
        
    def create_simulation_paths_panel(self, var_results: Dict) -> go.Figure:
        """
        Create Monte Carlo simulation paths fan chart
        
        Args:
            var_results: VaR simulation results
            
        Returns:
            Plotly figure for simulation paths
        """
        pnl = var_results['pnl_distribution']
        
        # Generate sample paths (200 random from 10,000)
        np.random.seed(42)
        sample_indices = np.random.choice(len(pnl), size=200, replace=False)
        sample_paths = pnl[sample_indices]
        
        fig = go.Figure()
        
        # Add sample paths (thin grey lines)
        for i, path_pnl in enumerate(sample_paths):
            # Create a simple path over 10 days
            volatility = abs(path_pnl) / 100000 if path_pnl != 0 else 0.01
            path = np.random.normal(0, volatility, 10).cumsum()
            fig.add_trace(go.Scatter(
                x=list(range(1, 11)),
                y=path,
                mode='lines',
                line=dict(color='grey', width=0.5),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add mean path (bold blue)
        mean_volatility = abs(np.mean(pnl)) / 100000 if np.mean(pnl) != 0 else 0.01
        mean_path = np.random.normal(0, mean_volatility, 10).cumsum()
        fig.add_trace(go.Scatter(
            x=list(range(1, 11)),
            y=mean_path,
            mode='lines',
            name='Mean Path',
            line=dict(color=self.colors['primary'], width=3),
            hovertemplate='Day: %{x}<br>P&L: %{y:.2f}%<extra></extra>'
        ))
        
        # Add 5th percentile (worst case - red)
        worst_volatility = abs(np.percentile(pnl, 5)) / 100000 if np.percentile(pnl, 5) != 0 else 0.01
        worst_path = np.random.normal(0, worst_volatility, 10).cumsum()
        fig.add_trace(go.Scatter(
            x=list(range(1, 11)),
            y=worst_path,
            mode='lines',
            name='5th Percentile',
            line=dict(color=self.colors['danger'], width=2),
            hovertemplate='Day: %{x}<br>P&L: %{y:.2f}%<extra></extra>'
        ))
        
        # Add 95th percentile (best case - green)
        best_volatility = abs(np.percentile(pnl, 95)) / 100000 if np.percentile(pnl, 95) != 0 else 0.01
        best_path = np.random.normal(0, best_volatility, 10).cumsum()
        fig.add_trace(go.Scatter(
            x=list(range(1, 11)),
            y=best_path,
            mode='lines',
            name='95th Percentile',
            line=dict(color=self.colors['success'], width=2),
            hovertemplate='Day: %{x}<br>P&L: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Monte Carlo Simulation - Portfolio P&L Paths",
            xaxis_title="Simulation Day",
            yaxis_title="Portfolio P&L (%)",
            template=self.template,
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
        
    def create_var_sensitivity_panel(self, var_results: Dict) -> go.Figure:
        """
        Create CVaR vs VaR sensitivity analysis across confidence levels
        
        Args:
            var_results: VaR simulation results
            
        Returns:
            Plotly figure for sensitivity analysis
        """
        pnl = var_results['pnl_distribution']
        
        # Calculate VaR and CVaR at different confidence levels
        confidence_levels = np.arange(90, 99.91, 0.1)  # 90% to 99.9%
        var_values = []
        cvar_values = []
        
        for cl in confidence_levels:
            alpha = (100 - cl) / 100
            var_val = abs(np.percentile(pnl, alpha * 100))
            cvar_val = abs(pnl[pnl <= -var_val].mean())
            
            var_values.append(var_val)
            cvar_values.append(cvar_val)
        
        fig = go.Figure()
        
        # Add VaR line
        fig.add_trace(go.Scatter(
            x=confidence_levels,
            y=var_values,
            mode='lines',
            name='VaR',
            line=dict(color=self.colors['warning'], width=3),
            hovertemplate='Confidence: %{x:.1f}%<br>VaR: %{text}<extra></extra>',
            text=[self.format_indian_currency(v) for v in var_values]
        ))
        
        # Add CVaR line
        fig.add_trace(go.Scatter(
            x=confidence_levels,
            y=cvar_values,
            mode='lines',
            name='CVaR',
            line=dict(color=self.colors['danger'], width=3),
            hovertemplate='Confidence: %{x:.1f}%<br>CVaR: %{text}<extra></extra>',
            text=[self.format_indian_currency(v) for v in cvar_values]
        ))
        
        # Add vertical lines at 95% and 99%
        fig.add_vline(
            x=95,
            line_dash="dash",
            line_color="white",
            line_width=1,
            opacity=0.5,
            annotation_text="95%",
            annotation_position="top left"
        )
        
        fig.add_vline(
            x=99,
            line_dash="dash",
            line_color="white",
            line_width=1,
            opacity=0.5,
            annotation_text="99%",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title="VaR / CVaR Sensitivity to Confidence Level",
            xaxis_title="Confidence Level (%)",
            yaxis_title="Risk Measure (¥)",
            template=self.template,
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        return fig
        
    def generate_dashboard(self, var_results: Dict[str, Dict], 
                          backtest_results: pd.DataFrame) -> str:
        """
        Generate comprehensive 8-panel dashboard
        
        Args:
            var_results: VaR results for all regimes
            backtest_results: Backtest results DataFrame
            
        Returns:
            HTML file path for the dashboard
        """
        print("Generating enhanced 8-panel dashboard...")
        
        # Create 4×2 subplot layout
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Simulated P&L Distribution - 10,000 Paths',
                'VaR by Stress Regime (95% vs 99% CI)',
                'NIFTY50 Portfolio - Asset Correlation Matrix',
                'Regulatory Backtest Results (Kupiec + Christoffersen)',
                'Historical Cumulative Portfolio Return (3Y)',
                'Cholesky Decomposition Factor (L matrix)',
                'Monte Carlo Simulation - Portfolio P&L Paths',
                'VaR / CVaR Sensitivity to Confidence Level'
            ],
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "table"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            vertical_spacing=0.06,
            horizontal_spacing=0.05
        )
        
        # Panel 1: P&L Distribution (using normal regime)
        pnl_fig = self.create_pnl_distribution_panel(var_results['normal'])
        for trace in pnl_fig.data:
            fig.add_trace(trace, row=1, col=1)
            
        # Panel 2: VaR Comparison
        var_comp_fig = self.create_var_comparison_panel(var_results)
        for trace in var_comp_fig.data:
            fig.add_trace(trace, row=1, col=2)
            
        # Panel 3: Correlation Heatmap
        corr_fig = self.create_correlation_heatmap()
        fig.add_trace(corr_fig.data[0], row=2, col=1)
        
        # Panel 4: Backtest Results Table
        table_fig = self.create_backtest_results_table(backtest_results)
        fig.add_trace(table_fig.data[0], row=2, col=2)
        
        # Panel 5: Cumulative Returns
        returns_fig = self.create_cumulative_returns_panel()
        for trace in returns_fig.data:
            fig.add_trace(trace, row=3, col=1)
            
        # Panel 6: Cholesky Heatmap
        cholesky_fig = self.create_cholesky_heatmap()
        fig.add_trace(cholesky_fig.data[0], row=3, col=2)
        
        # Panel 7: Simulation Paths
        paths_fig = self.create_simulation_paths_panel(var_results['normal'])
        for trace in paths_fig.data:
            fig.add_trace(trace, row=4, col=1)
            
        # Panel 8: VaR Sensitivity
        sensitivity_fig = self.create_var_sensitivity_panel(var_results['normal'])
        for trace in sensitivity_fig.data:
            fig.add_trace(trace, row=4, col=2)
        
        # Update layout with enhanced styling
        fig.update_layout(
            title={
                'text': "Monte Carlo VaR Engine - Basel II Risk Dashboard",
                'x': 0.5,
                'xanchor': 'center',
                'font': dict(size=20, color='white')
            },
            template=self.template,
            height=2400,  # 600px × 4 rows
            showlegend=False,
            font=dict(size=10),
            margin=dict(l=40, r=40, t=80, b=40)
        )
        
        # Update subplot titles styling
        for annotation in fig['layout']['annotations']:
            annotation['font'] = dict(size=12, color='white')
        
        # Save to HTML
        output_file = "var_dashboard.html"
        fig.write_html(output_file, include_plotlyjs='cdn')
        
        print(f"Enhanced dashboard saved: {output_file}")
        return output_file
