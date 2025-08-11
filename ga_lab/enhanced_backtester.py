# ga_lab/enhanced_backtester.py
"""
Enhanced backtesting framework with Monte Carlo analysis, performance attribution,
and comprehensive risk metrics for the genetic algorithm trading system.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from .strategy import GeneticStrategy
from .indicators import Indicators


@dataclass
class TradeRecord:
    """Detailed record of a single trade."""

    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    direction: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    duration_hours: float
    exit_reason: str  # 'signal', 'stop_loss', 'take_profit', 'end_of_data'
    max_favorable_excursion: float  # MFE
    max_adverse_excursion: float  # MAE


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Return metrics
    total_return: float
    annualized_return: float
    compound_annual_growth_rate: float

    # Risk metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    value_at_risk_95: float
    conditional_value_at_risk_95: float

    # Trade metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float

    # Advanced metrics
    kelly_criterion: float
    optimal_f: float
    recovery_factor: float
    sterling_ratio: float
    burke_ratio: float

    # Market exposure
    time_in_market: float
    long_trades: int
    short_trades: int


class EnhancedBacktester:
    """Enhanced backtesting system with comprehensive analysis."""

    def __init__(self, cfg: Dict, logger: logging.Logger):
        self.cfg = cfg
        self.log = logger

        # Backtesting configuration
        bt_cfg = cfg.get("enhanced_backtesting", {})
        self.monte_carlo_runs = bt_cfg.get("monte_carlo_runs", 1000)
        self.confidence_levels = bt_cfg.get("confidence_levels", [0.05, 0.95])
        self.bootstrap_samples = bt_cfg.get("bootstrap_samples", 500)
        self.enable_detailed_analysis = bt_cfg.get("enable_detailed_analysis", True)

    def run_comprehensive_backtest(
        self, strategy: GeneticStrategy, candles: pd.DataFrame, initial_balance: float = 10000.0
    ) -> Dict:
        """
        Run comprehensive backtest with detailed analysis.

        Returns:
            Dictionary containing all analysis results
        """
        self.log.info(f"Running comprehensive backtest for strategy {strategy.id}")

        # Run detailed simulation
        simulation_result = self._run_detailed_simulation(strategy, candles, initial_balance)

        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            simulation_result["trades"], simulation_result["equity_curve"], initial_balance
        )

        # Monte Carlo analysis
        monte_carlo_results = self._run_monte_carlo_analysis(simulation_result["trades"], initial_balance)

        # Bootstrap analysis
        bootstrap_results = self._run_bootstrap_analysis(simulation_result["trades"], initial_balance)

        # Performance attribution
        attribution = self._analyze_performance_attribution(simulation_result["trades"], candles)

        # Risk analysis
        risk_analysis = self._analyze_risk_metrics(simulation_result["equity_curve"], simulation_result["trades"])

        return {
            "strategy_id": strategy.id,
            "simulation": simulation_result,
            "metrics": metrics,
            "monte_carlo": monte_carlo_results,
            "bootstrap": bootstrap_results,
            "attribution": attribution,
            "risk_analysis": risk_analysis,
            "summary": self._generate_summary(metrics, monte_carlo_results),
        }

    def _run_detailed_simulation(
        self, strategy: GeneticStrategy, candles: pd.DataFrame, initial_balance: float
    ) -> Dict:
        """Run detailed simulation tracking all trades and equity."""
        # Add indicators
        ind_df = Indicators.add_all(candles, required_params=strategy.indicator_params)

        balance = initial_balance
        position_size = 0.0
        entry_price = 0.0
        entry_time = None
        position_direction = None

        trades = []
        equity_curve = [initial_balance]
        daily_returns = []

        # Simulation configuration
        sim_cfg = self.cfg["simulation"]
        transaction_cost = float(sim_cfg.get("transaction_cost_pct", 0.001))
        slippage = float(sim_cfg.get("slippage_pct", 0.0005))

        for i in range(len(ind_df)):
            current_row = ind_df.iloc[i]
            price = candles.iloc[i]["close"]
            high = candles.iloc[i]["high"]
            low = candles.iloc[i]["low"]
            timestamp = candles.index[i] if hasattr(candles.index[i], "to_pydatetime") else datetime.now()

            # Calculate current equity
            if position_size != 0:
                unrealized_pnl = position_size * (price - entry_price)
                current_equity = balance + unrealized_pnl
            else:
                current_equity = balance

            equity_curve.append(current_equity)

            # Calculate daily return
            if len(equity_curve) > 1:
                daily_return = (equity_curve[-1] / equity_curve[-2]) - 1
                daily_returns.append(daily_return)

            # Get trading signal
            action, confidence = strategy.evaluate(current_row)

            # Process trading signals
            if action == "buy" and position_size <= 0:
                # Close short position if exists
                if position_size < 0:
                    pnl = position_size * (price - entry_price)
                    net_pnl = pnl - abs(pnl * transaction_cost) - abs(position_size * price * slippage)
                    balance += net_pnl

                    # Record trade
                    trade = TradeRecord(
                        entry_time=entry_time,
                        exit_time=timestamp,
                        entry_price=entry_price,
                        exit_price=price,
                        position_size=abs(position_size),
                        direction="short",
                        pnl=net_pnl,
                        pnl_pct=net_pnl / (abs(position_size) * entry_price),
                        duration_hours=(timestamp - entry_time).total_seconds() / 3600 if entry_time else 0,
                        exit_reason="signal",
                        max_favorable_excursion=0,  # Would need to track during position
                        max_adverse_excursion=0,  # Would need to track during position
                    )
                    trades.append(trade)

                # Open long position
                if balance > 0:
                    position_fraction = min(0.95, confidence)  # Use confidence for position sizing
                    position_size = (balance * position_fraction) / price
                    entry_price = price * (1 + slippage)
                    entry_time = timestamp
                    position_direction = "long"

            elif action == "sell" and position_size >= 0:
                # Close long position if exists
                if position_size > 0:
                    pnl = position_size * (price - entry_price)
                    net_pnl = pnl - abs(pnl * transaction_cost) - abs(position_size * price * slippage)
                    balance += net_pnl

                    # Record trade
                    trade = TradeRecord(
                        entry_time=entry_time,
                        exit_time=timestamp,
                        entry_price=entry_price,
                        exit_price=price,
                        position_size=position_size,
                        direction="long",
                        pnl=net_pnl,
                        pnl_pct=net_pnl / (position_size * entry_price),
                        duration_hours=(timestamp - entry_time).total_seconds() / 3600 if entry_time else 0,
                        exit_reason="signal",
                        max_favorable_excursion=0,
                        max_adverse_excursion=0,
                    )
                    trades.append(trade)

                # Open short position
                if balance > 0:
                    position_fraction = min(0.95, confidence)
                    position_size = -(balance * position_fraction) / price
                    entry_price = price * (1 - slippage)
                    entry_time = timestamp
                    position_direction = "short"

        # Close final position if exists
        if position_size != 0:
            final_price = candles.iloc[-1]["close"]
            pnl = position_size * (final_price - entry_price)
            net_pnl = pnl - abs(pnl * transaction_cost) - abs(position_size * final_price * slippage)
            balance += net_pnl

            trade = TradeRecord(
                entry_time=entry_time,
                exit_time=candles.index[-1] if hasattr(candles.index[-1], "to_pydatetime") else datetime.now(),
                entry_price=entry_price,
                exit_price=final_price,
                position_size=abs(position_size),
                direction=position_direction,
                pnl=net_pnl,
                pnl_pct=net_pnl / (abs(position_size) * entry_price),
                duration_hours=0,
                exit_reason="end_of_data",
                max_favorable_excursion=0,
                max_adverse_excursion=0,
            )
            trades.append(trade)

        return {
            "trades": trades,
            "equity_curve": equity_curve,
            "daily_returns": daily_returns,
            "final_balance": balance,
        }

    def _calculate_comprehensive_metrics(
        self, trades: List[TradeRecord], equity_curve: List[float], initial_balance: float
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return PerformanceMetrics(
                total_return=0,
                annualized_return=0,
                compound_annual_growth_rate=0,
                volatility=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                calmar_ratio=0,
                max_drawdown=0,
                max_drawdown_duration=0,
                value_at_risk_95=0,
                conditional_value_at_risk_95=0,
                total_trades=0,
                win_rate=0,
                profit_factor=0,
                avg_win=0,
                avg_loss=0,
                largest_win=0,
                largest_loss=0,
                avg_trade_duration=0,
                kelly_criterion=0,
                optimal_f=0,
                recovery_factor=0,
                sterling_ratio=0,
                burke_ratio=0,
                time_in_market=0,
                long_trades=0,
                short_trades=0,
            )

        # Basic calculations
        final_balance = equity_curve[-1]
        total_return = (final_balance - initial_balance) / initial_balance

        # Trade statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        profit_factor = (
            abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades))
            if losing_trades
            else float("inf")
        )

        # Risk metrics
        returns = [(equity_curve[i] / equity_curve[i - 1] - 1) for i in range(1, len(equity_curve))]
        volatility = np.std(returns) * np.sqrt(252) if returns else 0

        # Drawdown calculation
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peak) / peak
        max_drawdown = abs(np.min(drawdown))

        # Sharpe ratio
        sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if returns and np.std(returns) > 0 else 0

        # Sortino ratio
        negative_returns = [r for r in returns if r < 0]
        downside_deviation = np.std(negative_returns) if negative_returns else 0
        sortino_ratio = (np.mean(returns) / downside_deviation * np.sqrt(252)) if downside_deviation > 0 else 0

        # Calmar ratio
        annualized_return = (
            (final_balance / initial_balance) ** (252 / len(equity_curve)) - 1
            if len(equity_curve) > 252
            else total_return
        )
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            compound_annual_growth_rate=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=0,  # Would need to calculate
            value_at_risk_95=np.percentile(returns, 5) if returns else 0,
            conditional_value_at_risk_95=(
                np.mean([r for r in returns if r <= np.percentile(returns, 5)]) if returns else 0
            ),
            total_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=max([t.pnl for t in trades]) if trades else 0,
            largest_loss=min([t.pnl for t in trades]) if trades else 0,
            avg_trade_duration=np.mean([t.duration_hours for t in trades]) if trades else 0,
            kelly_criterion=0,  # Would need to calculate
            optimal_f=0,  # Would need to calculate
            recovery_factor=total_return / max_drawdown if max_drawdown > 0 else 0,
            sterling_ratio=0,  # Would need to calculate
            burke_ratio=0,  # Would need to calculate
            time_in_market=0,  # Would need to calculate
            long_trades=len([t for t in trades if t.direction == "long"]),
            short_trades=len([t for t in trades if t.direction == "short"]),
        )

    def _run_monte_carlo_analysis(self, trades: List[TradeRecord], initial_balance: float) -> Dict:
        """Run Monte Carlo analysis by randomizing trade order."""
        if not trades:
            return {"error": "No trades for Monte Carlo analysis"}

        trade_returns = [t.pnl / initial_balance for t in trades]
        mc_results = []

        for _ in range(self.monte_carlo_runs):
            # Randomize trade order
            shuffled_returns = np.random.choice(trade_returns, len(trade_returns), replace=True)

            # Calculate cumulative return
            cumulative_balance = initial_balance
            equity_path = [cumulative_balance]

            for ret in shuffled_returns:
                cumulative_balance *= 1 + ret
                equity_path.append(cumulative_balance)

            # Calculate metrics for this run
            final_return = (cumulative_balance - initial_balance) / initial_balance

            # Calculate max drawdown
            peak = np.maximum.accumulate(equity_path)
            drawdown = (np.array(equity_path) - peak) / peak
            max_dd = abs(np.min(drawdown))

            mc_results.append(
                {"final_return": final_return, "max_drawdown": max_dd, "final_balance": cumulative_balance}
            )

        # Analyze Monte Carlo results
        returns = [r["final_return"] for r in mc_results]
        drawdowns = [r["max_drawdown"] for r in mc_results]

        return {
            "runs": len(mc_results),
            "return_statistics": {
                "mean": np.mean(returns),
                "std": np.std(returns),
                "min": np.min(returns),
                "max": np.max(returns),
                "percentiles": {
                    "5th": np.percentile(returns, 5),
                    "25th": np.percentile(returns, 25),
                    "50th": np.percentile(returns, 50),
                    "75th": np.percentile(returns, 75),
                    "95th": np.percentile(returns, 95),
                },
            },
            "drawdown_statistics": {
                "mean": np.mean(drawdowns),
                "std": np.std(drawdowns),
                "min": np.min(drawdowns),
                "max": np.max(drawdowns),
                "percentiles": {
                    "5th": np.percentile(drawdowns, 5),
                    "25th": np.percentile(drawdowns, 25),
                    "50th": np.percentile(drawdowns, 50),
                    "75th": np.percentile(drawdowns, 75),
                    "95th": np.percentile(drawdowns, 95),
                },
            },
            "probability_of_loss": sum(1 for r in returns if r < 0) / len(returns),
            "probability_of_large_drawdown": sum(1 for dd in drawdowns if dd > 0.2) / len(drawdowns),
        }

    def _run_bootstrap_analysis(self, trades: List[TradeRecord], initial_balance: float) -> Dict:
        """Run bootstrap analysis for confidence intervals."""
        if not trades:
            return {"error": "No trades for bootstrap analysis"}

        trade_returns = [t.pnl / initial_balance for t in trades]
        bootstrap_results = []

        for _ in range(self.bootstrap_samples):
            # Bootstrap sample with replacement
            sample_returns = np.random.choice(trade_returns, len(trade_returns), replace=True)

            # Calculate sample statistics
            sample_mean = np.mean(sample_returns)
            sample_std = np.std(sample_returns)
            sample_sharpe = sample_mean / sample_std * np.sqrt(len(sample_returns)) if sample_std > 0 else 0

            bootstrap_results.append(
                {"mean_return": sample_mean, "std_return": sample_std, "sharpe_ratio": sample_sharpe}
            )

        # Calculate confidence intervals
        mean_returns = [r["mean_return"] for r in bootstrap_results]
        sharpe_ratios = [r["sharpe_ratio"] for r in bootstrap_results]

        return {
            "samples": len(bootstrap_results),
            "mean_return_ci": {
                "lower_95": np.percentile(mean_returns, 2.5),
                "upper_95": np.percentile(mean_returns, 97.5),
                "lower_90": np.percentile(mean_returns, 5),
                "upper_90": np.percentile(mean_returns, 95),
            },
            "sharpe_ratio_ci": {
                "lower_95": np.percentile(sharpe_ratios, 2.5),
                "upper_95": np.percentile(sharpe_ratios, 97.5),
                "lower_90": np.percentile(sharpe_ratios, 5),
                "upper_90": np.percentile(sharpe_ratios, 95),
            },
        }

    def _analyze_performance_attribution(self, trades: List[TradeRecord], candles: pd.DataFrame) -> Dict:
        """Analyze performance attribution by various factors."""
        if not trades:
            return {"error": "No trades for attribution analysis"}

        # Analyze by trade direction
        long_trades = [t for t in trades if t.direction == "long"]
        short_trades = [t for t in trades if t.direction == "short"]

        # Analyze by time periods (if we have enough data)
        attribution = {
            "by_direction": {
                "long": {
                    "count": len(long_trades),
                    "total_pnl": sum(t.pnl for t in long_trades),
                    "avg_pnl": np.mean([t.pnl for t in long_trades]) if long_trades else 0,
                    "win_rate": sum(1 for t in long_trades if t.pnl > 0) / len(long_trades) if long_trades else 0,
                },
                "short": {
                    "count": len(short_trades),
                    "total_pnl": sum(t.pnl for t in short_trades),
                    "avg_pnl": np.mean([t.pnl for t in short_trades]) if short_trades else 0,
                    "win_rate": sum(1 for t in short_trades if t.pnl > 0) / len(short_trades) if short_trades else 0,
                },
            },
            "by_duration": self._analyze_by_duration(trades),
            "by_exit_reason": self._analyze_by_exit_reason(trades),
        }

        return attribution

    def _analyze_by_duration(self, trades: List[TradeRecord]) -> Dict:
        """Analyze performance by trade duration."""
        if not trades:
            return {}

        # Categorize trades by duration
        short_term = [t for t in trades if t.duration_hours <= 24]  # Less than 1 day
        medium_term = [t for t in trades if 24 < t.duration_hours <= 168]  # 1-7 days
        long_term = [t for t in trades if t.duration_hours > 168]  # More than 7 days

        return {
            "short_term": {
                "count": len(short_term),
                "avg_pnl": np.mean([t.pnl for t in short_term]) if short_term else 0,
                "win_rate": sum(1 for t in short_term if t.pnl > 0) / len(short_term) if short_term else 0,
            },
            "medium_term": {
                "count": len(medium_term),
                "avg_pnl": np.mean([t.pnl for t in medium_term]) if medium_term else 0,
                "win_rate": sum(1 for t in medium_term if t.pnl > 0) / len(medium_term) if medium_term else 0,
            },
            "long_term": {
                "count": len(long_term),
                "avg_pnl": np.mean([t.pnl for t in long_term]) if long_term else 0,
                "win_rate": sum(1 for t in long_term if t.pnl > 0) / len(long_term) if long_term else 0,
            },
        }

    def _analyze_by_exit_reason(self, trades: List[TradeRecord]) -> Dict:
        """Analyze performance by exit reason."""
        exit_reasons = {}

        for trade in trades:
            reason = trade.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = []
            exit_reasons[reason].append(trade)

        analysis = {}
        for reason, reason_trades in exit_reasons.items():
            analysis[reason] = {
                "count": len(reason_trades),
                "total_pnl": sum(t.pnl for t in reason_trades),
                "avg_pnl": np.mean([t.pnl for t in reason_trades]),
                "win_rate": sum(1 for t in reason_trades if t.pnl > 0) / len(reason_trades),
            }

        return analysis

    def _analyze_risk_metrics(self, equity_curve: List[float], trades: List[TradeRecord]) -> Dict:
        """Analyze detailed risk metrics."""
        if len(equity_curve) < 2:
            return {"error": "Insufficient data for risk analysis"}

        returns = [(equity_curve[i] / equity_curve[i - 1] - 1) for i in range(1, len(equity_curve))]

        # Calculate various risk metrics
        return {
            "volatility_analysis": {
                "daily_volatility": np.std(returns),
                "annualized_volatility": np.std(returns) * np.sqrt(252),
                "volatility_of_volatility": (
                    np.std([np.std(returns[i : i + 20]) for i in range(len(returns) - 20)]) if len(returns) > 20 else 0
                ),
            },
            "tail_risk": {
                "skewness": self._calculate_skewness(returns),
                "kurtosis": self._calculate_kurtosis(returns),
                "var_95": np.percentile(returns, 5),
                "cvar_95": np.mean([r for r in returns if r <= np.percentile(returns, 5)]) if returns else 0,
            },
            "drawdown_analysis": self._analyze_drawdowns(equity_curve),
            "correlation_analysis": self._analyze_trade_correlations(trades),
        }

    def _calculate_skewness(self, returns: List[float]) -> float:
        """Calculate skewness of returns."""
        if len(returns) < 3:
            return 0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0

        skewness = np.mean([((r - mean_return) / std_return) ** 3 for r in returns])
        return skewness

    def _calculate_kurtosis(self, returns: List[float]) -> float:
        """Calculate kurtosis of returns."""
        if len(returns) < 4:
            return 0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0

        kurtosis = np.mean([((r - mean_return) / std_return) ** 4 for r in returns]) - 3
        return kurtosis

    def _analyze_drawdowns(self, equity_curve: List[float]) -> Dict:
        """Analyze drawdown characteristics."""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peak) / peak

        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []

        start_idx = None
        for i, is_dd in enumerate(in_drawdown):
            if is_dd and start_idx is None:
                start_idx = i
            elif not is_dd and start_idx is not None:
                drawdown_periods.append((start_idx, i - 1, np.min(drawdown[start_idx:i])))
                start_idx = None

        if start_idx is not None:  # Still in drawdown at end
            drawdown_periods.append((start_idx, len(drawdown) - 1, np.min(drawdown[start_idx:])))

        return {
            "max_drawdown": abs(np.min(drawdown)),
            "avg_drawdown": np.mean([abs(dd[2]) for dd in drawdown_periods]) if drawdown_periods else 0,
            "drawdown_periods": len(drawdown_periods),
            "avg_drawdown_duration": np.mean([dd[1] - dd[0] + 1 for dd in drawdown_periods]) if drawdown_periods else 0,
            "max_drawdown_duration": max([dd[1] - dd[0] + 1 for dd in drawdown_periods]) if drawdown_periods else 0,
        }

    def _analyze_trade_correlations(self, trades: List[TradeRecord]) -> Dict:
        """Analyze correlations between consecutive trades."""
        if len(trades) < 2:
            return {"error": "Insufficient trades for correlation analysis"}

        returns = [t.pnl_pct for t in trades]

        # Serial correlation
        serial_corr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0

        return {"serial_correlation": serial_corr, "runs_test": self._runs_test(returns)}

    def _runs_test(self, returns: List[float]) -> Dict:
        """Simple runs test for randomness."""
        if len(returns) < 2:
            return {"error": "Insufficient data"}

        # Convert to binary sequence (positive/negative)
        binary_seq = [1 if r > 0 else 0 for r in returns]

        # Count runs
        runs = 1
        for i in range(1, len(binary_seq)):
            if binary_seq[i] != binary_seq[i - 1]:
                runs += 1

        # Expected runs under null hypothesis of randomness
        n1 = sum(binary_seq)  # Number of positive returns
        n2 = len(binary_seq) - n1  # Number of negative returns

        if n1 == 0 or n2 == 0:
            return {"runs": runs, "expected_runs": 0, "z_score": 0}

        expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))

        z_score = (runs - expected_runs) / np.sqrt(variance) if variance > 0 else 0

        return {
            "runs": runs,
            "expected_runs": expected_runs,
            "z_score": z_score,
            "is_random": abs(z_score) < 1.96,  # 95% confidence level
        }

    def _generate_summary(self, metrics: PerformanceMetrics, monte_carlo: Dict) -> Dict:
        """Generate executive summary of backtest results."""
        return {
            "overall_rating": self._calculate_overall_rating(metrics),
            "key_strengths": self._identify_strengths(metrics),
            "key_weaknesses": self._identify_weaknesses(metrics),
            "risk_assessment": self._assess_risk_level(metrics, monte_carlo),
            "recommendations": self._generate_recommendations(metrics, monte_carlo),
        }

    def _calculate_overall_rating(self, metrics: PerformanceMetrics) -> str:
        """Calculate overall strategy rating."""
        score = 0

        # Return component (0-30 points)
        if metrics.annualized_return > 0.2:
            score += 30
        elif metrics.annualized_return > 0.1:
            score += 20
        elif metrics.annualized_return > 0.05:
            score += 10

        # Risk component (0-30 points)
        if metrics.sharpe_ratio > 2.0:
            score += 30
        elif metrics.sharpe_ratio > 1.0:
            score += 20
        elif metrics.sharpe_ratio > 0.5:
            score += 10

        # Drawdown component (0-25 points)
        if metrics.max_drawdown < 0.05:
            score += 25
        elif metrics.max_drawdown < 0.1:
            score += 20
        elif metrics.max_drawdown < 0.2:
            score += 15
        elif metrics.max_drawdown < 0.3:
            score += 10

        # Consistency component (0-15 points)
        if metrics.win_rate > 0.6:
            score += 15
        elif metrics.win_rate > 0.5:
            score += 10
        elif metrics.win_rate > 0.4:
            score += 5

        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        elif score >= 20:
            return "Poor"
        else:
            return "Very Poor"

    def _identify_strengths(self, metrics: PerformanceMetrics) -> List[str]:
        """Identify strategy strengths."""
        strengths = []

        if metrics.sharpe_ratio > 1.5:
            strengths.append("Excellent risk-adjusted returns")
        if metrics.max_drawdown < 0.1:
            strengths.append("Low maximum drawdown")
        if metrics.win_rate > 0.6:
            strengths.append("High win rate")
        if metrics.profit_factor > 2.0:
            strengths.append("Strong profit factor")
        if metrics.calmar_ratio > 1.0:
            strengths.append("Good return-to-drawdown ratio")

        return strengths

    def _identify_weaknesses(self, metrics: PerformanceMetrics) -> List[str]:
        """Identify strategy weaknesses."""
        weaknesses = []

        if metrics.sharpe_ratio < 0.5:
            weaknesses.append("Poor risk-adjusted returns")
        if metrics.max_drawdown > 0.3:
            weaknesses.append("High maximum drawdown")
        if metrics.win_rate < 0.4:
            weaknesses.append("Low win rate")
        if metrics.profit_factor < 1.2:
            weaknesses.append("Weak profit factor")
        if metrics.total_trades < 20:
            weaknesses.append("Insufficient trade frequency")

        return weaknesses

    def _assess_risk_level(self, metrics: PerformanceMetrics, monte_carlo: Dict) -> str:
        """Assess overall risk level."""
        risk_score = 0

        if metrics.max_drawdown > 0.3:
            risk_score += 3
        elif metrics.max_drawdown > 0.2:
            risk_score += 2
        elif metrics.max_drawdown > 0.1:
            risk_score += 1

        if metrics.volatility > 0.3:
            risk_score += 2
        elif metrics.volatility > 0.2:
            risk_score += 1

        if "probability_of_large_drawdown" in monte_carlo and monte_carlo["probability_of_large_drawdown"] > 0.2:
            risk_score += 2

        if risk_score >= 5:
            return "High Risk"
        elif risk_score >= 3:
            return "Medium Risk"
        else:
            return "Low Risk"

    def _generate_recommendations(self, metrics: PerformanceMetrics, monte_carlo: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if metrics.max_drawdown > 0.2:
            recommendations.append("Consider implementing stricter risk management")
        if metrics.win_rate < 0.5:
            recommendations.append("Review entry criteria to improve win rate")
        if metrics.total_trades < 30:
            recommendations.append("Consider increasing trade frequency for better statistical significance")
        if metrics.sharpe_ratio < 1.0:
            recommendations.append("Focus on improving risk-adjusted returns")

        return recommendations
