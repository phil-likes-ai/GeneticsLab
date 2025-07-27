from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .indicators import Indicators
from .strategy import GeneticStrategy


class Backtester:
    """Runs a single strategy against historical data and calculates performance metrics."""

    @staticmethod
    def plot_equity_curve(equity_curve: list[float], candles: pd.DataFrame, strategy_id: str) -> str:
        """Plots the equity curve and saves it to a file."""
        fig, ax = plt.subplots(figsize=(15, 7))

        # Ensure the equity curve and candle index have the same length for plotting
        if len(equity_curve) > len(candles.index):
            equity_curve = equity_curve[:len(candles.index)]
        elif len(equity_curve) < len(candles.index):
            candles = candles.iloc[:len(equity_curve)]

        ax.plot(candles.index, equity_curve, label="Equity Curve", color="#007bff")
        ax.fill_between(candles.index, equity_curve, alpha=0.1, color="#007bff")

        ax.set_title(f"Equity Curve for Strategy: {strategy_id}", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Formatting the x-axis for dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        # Add a starting balance line
        ax.axhline(y=candles.iloc[0]['open'], color='r', linestyle='--', label=f'Starting Price: ${candles.iloc[0]["open"]:,.2f}')

        plt.legend()
        plt.tight_layout()

        # Save the plot
        plot_filename = f"backtest_plot_{strategy_id}.png"
        plt.savefig(plot_filename)
        plt.close(fig)  # Close the figure to free memory

        return plot_filename

    @staticmethod
    def run(
        strategy: GeneticStrategy, candles: pd.DataFrame, initial_balance: float = 1000.0, generate_plot: bool = False
    ) -> dict:
        """
        Simulates the strategy and returns a dictionary of performance metrics.
        """
        if candles.empty:
            return {}

        ind_df = Indicators.add_all(candles.copy())

        cash = initial_balance
        asset_qty = 0.0
        trades = []
        equity_curve = []

        for i in range(len(ind_df)):
            price = ind_df.iloc[i]["close"]
            equity = cash + (asset_qty * price)
            equity_curve.append(equity)

            if equity <= 0:
                equity_curve.extend([0] * (len(ind_df) - len(equity_curve)))
                break

            action, _ = strategy.evaluate(ind_df.iloc[i])

            # --- Position Management ---
            # Close Long or Open Short
            if action == "sell":
                if asset_qty > 0: # Close existing long
                    cash += asset_qty * price
                    trades.append(cash - initial_balance) # Simple PnL for now
                    asset_qty = 0
                # You could add logic here to open a short if asset_qty is 0

            # Close Short or Open Long
            elif action == "buy":
                if asset_qty < 0: # Close existing short
                    cash += asset_qty * price
                    trades.append(cash - initial_balance)
                    asset_qty = 0
                elif asset_qty == 0 and cash > 0: # Open new long
                    asset_qty = cash / price
                    cash = 0

        # Final equity calculation
        if equity_curve and equity_curve[-1] > 0:
            final_equity = cash + (asset_qty * ind_df.iloc[-1]["close"])
            equity_curve.append(final_equity)

        metrics = Backtester.calculate_metrics(equity_curve, trades, initial_balance)

        if generate_plot:
            plot_file = Backtester.plot_equity_curve(equity_curve, candles, strategy.id)
            metrics["plot_file"] = plot_file

        return metrics

    @staticmethod
    def calculate_metrics(
        equity_curve: list[float], trades: list[float], initial_balance: float
    ) -> dict:
        if not trades:
            return {
                "total_return_pct": 0,
                "max_drawdown_pct": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "win_rate_pct": 0,
                "total_trades": 0,
                "profit_factor": 0,
            }

        equity_series = pd.Series(equity_curve)

        # Total Return
        total_return_pct = ((equity_series.iloc[-1] / initial_balance) - 1) * 100

        # Max Drawdown
        # A more robust way to calculate drawdown to avoid division issues
        peak = equity_series.cummax()
        if peak.iloc[-1] == 0 or peak.empty:
             max_drawdown_pct = -100.0
        else:
            drawdown = (equity_series / peak) - 1
            max_drawdown_pct = drawdown.min() * 100

        # Sharpe Ratio (assuming risk-free rate is 0 and daily returns)
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (
            (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
        )  # Annualized

        # Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std()
        sortino_ratio = (
            (returns.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0
        )  # Annualized

        # Win Rate
        wins = sum(1 for t in trades if t > 0)
        win_rate_pct = (wins / len(trades)) * 100

        # Profit Factor
        gross_profit = sum(t for t in trades if t > 0)
        gross_loss = abs(sum(t for t in trades if t < 0))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

        return {
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "win_rate_pct": round(win_rate_pct, 2),
            "total_trades": len(trades),
            "profit_factor": round(profit_factor, 2),
        }
