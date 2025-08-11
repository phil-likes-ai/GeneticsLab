# ga_lab/walk_forward.py
"""
Walk-forward validation system for the genetic algorithm trading system.
Implements rolling window optimization and out-of-sample testing to prevent overfitting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .manager import GeneticAlgorithmManager
from .strategy import GeneticStrategy


@dataclass
class WalkForwardResult:
    """Results from a single walk-forward window."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    best_strategy: GeneticStrategy
    train_fitness: float
    test_fitness: float
    test_return: float
    test_sharpe: float
    test_max_drawdown: float
    test_trades: int
    overfitting_ratio: float  # test_fitness / train_fitness


@dataclass
class WalkForwardSummary:
    """Summary of all walk-forward results."""

    total_windows: int
    avg_train_fitness: float
    avg_test_fitness: float
    avg_overfitting_ratio: float
    total_return: float
    total_sharpe: float
    max_drawdown: float
    win_rate: float
    stability_score: float  # Consistency across windows
    results: List[WalkForwardResult]


class WalkForwardValidator:
    """Walk-forward validation system for robust strategy optimization."""

    def __init__(self, cfg: Dict, logger: logging.Logger):
        self.cfg = cfg
        self.log = logger
        self.ga_manager = GeneticAlgorithmManager(cfg, logger)

        # Walk-forward configuration
        wf_cfg = cfg.get("walk_forward", {})
        self.train_window_days = wf_cfg.get("train_window_days", 252)  # 1 year
        self.test_window_days = wf_cfg.get("test_window_days", 63)  # 3 months
        self.step_days = wf_cfg.get("step_days", 21)  # 1 month step
        self.min_train_days = wf_cfg.get("min_train_days", 126)  # 6 months minimum
        self.max_overfitting_ratio = wf_cfg.get("max_overfitting_ratio", 0.5)
        self.stability_threshold = wf_cfg.get("stability_threshold", 0.7)

    def run_walk_forward_validation(
        self, db: "Database", symbol: str, timeframe: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None
    ) -> WalkForwardSummary:
        """
        Run complete walk-forward validation.

        Args:
            db: Database instance
            symbol: Trading symbol
            timeframe: Timeframe (e.g., '1d', '4h')
            start_ts: Start timestamp
            end_ts: End timestamp

        Returns:
            WalkForwardSummary with all results
        """
        # Load full dataset
        candles = db.load_candles(symbol=symbol, timeframe=timeframe, start_ts=start_ts, end_ts=end_ts)

        if candles.empty:
            self.log.error("No data available for walk-forward validation")
            return WalkForwardSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, [])

        # Generate walk-forward windows
        windows = self._generate_windows(candles)

        if not windows:
            self.log.error("No valid walk-forward windows generated")
            return WalkForwardSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, [])

        self.log.info(f"Running walk-forward validation with {len(windows)} windows")

        results = []
        combined_equity = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            self.log.info(f"Processing window {i+1}/{len(windows)}")

            # Train on training window
            train_data = candles.iloc[train_start : train_end + 1].copy()
            best_strategy = self._optimize_window(train_data, f"train_window_{i}")

            if best_strategy is None:
                self.log.warning(f"No valid strategy found for window {i+1}")
                continue

            # Test on out-of-sample window
            test_data = candles.iloc[test_start : test_end + 1].copy()
            test_result = self._evaluate_strategy(best_strategy, test_data)

            # Calculate overfitting ratio
            overfitting_ratio = test_result["fitness"] / best_strategy.fitness if best_strategy.fitness > 0 else 0

            # Store results
            wf_result = WalkForwardResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_strategy=best_strategy,
                train_fitness=best_strategy.fitness,
                test_fitness=test_result["fitness"],
                test_return=test_result["return"],
                test_sharpe=test_result["sharpe"],
                test_max_drawdown=test_result["max_drawdown"],
                test_trades=test_result["trades"],
                overfitting_ratio=overfitting_ratio,
            )

            results.append(wf_result)

            # Combine equity curves for overall performance
            if "equity_curve" in test_result:
                if not combined_equity:
                    combined_equity = test_result["equity_curve"]
                else:
                    # Normalize and combine
                    start_value = combined_equity[-1]
                    normalized_curve = [
                        start_value * (eq / test_result["equity_curve"][0]) for eq in test_result["equity_curve"]
                    ]
                    combined_equity.extend(normalized_curve[1:])  # Skip first to avoid duplication

        # Calculate summary statistics
        summary = self._calculate_summary(results, combined_equity)

        self.log.info(
            f"Walk-forward validation completed. "
            f"Avg overfitting ratio: {summary.avg_overfitting_ratio:.3f}, "
            f"Stability score: {summary.stability_score:.3f}"
        )

        return summary

    def _generate_windows(self, candles: pd.DataFrame) -> List[Tuple[int, int, int, int]]:
        """Generate walk-forward windows."""
        windows = []
        total_rows = len(candles)

        current_start = 0

        while current_start + self.min_train_days < total_rows:
            # Training window
            train_start = current_start
            train_end = min(current_start + self.train_window_days - 1, total_rows - self.test_window_days - 1)

            # Test window
            test_start = train_end + 1
            test_end = min(test_start + self.test_window_days - 1, total_rows - 1)

            # Validate window
            if (train_end - train_start + 1) >= self.min_train_days and test_end > test_start:
                windows.append((train_start, train_end, test_start, test_end))

            # Move to next window
            current_start += self.step_days

            # Stop if we can't create a valid test window
            if test_end >= total_rows - 1:
                break

        return windows

    def _optimize_window(self, train_data: pd.DataFrame, window_id: str) -> Optional[GeneticStrategy]:
        """Optimize strategy on training window."""
        try:
            # Create a temporary database-like object for the training data
            class TempDB:
                def __init__(self, data):
                    self.data = data

                def load_candles(self, **kwargs):
                    return self.data

            temp_db = TempDB(train_data)

            # Run genetic algorithm optimization
            strategies = self.ga_manager.evolve(db=temp_db, symbol="temp", timeframe="temp")

            if strategies:
                return strategies[0]  # Return best strategy

        except Exception as e:
            self.log.error(f"Error optimizing window {window_id}: {e}")

        return None

    def _evaluate_strategy(self, strategy: GeneticStrategy, test_data: pd.DataFrame) -> Dict:
        """Evaluate strategy on test data."""
        try:
            # Simulate strategy on test data
            balance = self.ga_manager._simulate(strategy, test_data)

            # Calculate additional metrics
            initial_balance = float(self.cfg["simulation"]["initial_balance"])
            total_return = (balance - initial_balance) / initial_balance

            # Calculate Sharpe ratio from equity curve
            if hasattr(strategy, "equity_curve") and len(strategy.equity_curve) > 1:
                returns = [
                    (strategy.equity_curve[i] / strategy.equity_curve[i - 1] - 1)
                    for i in range(1, len(strategy.equity_curve))
                ]
                sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252) if returns else 0
            else:
                sharpe = 0

            return {
                "fitness": strategy.fitness,
                "return": total_return,
                "sharpe": sharpe,
                "max_drawdown": getattr(strategy, "max_drawdown", 0),
                "trades": strategy.trades,
                "equity_curve": getattr(strategy, "equity_curve", []),
            }

        except Exception as e:
            self.log.error(f"Error evaluating strategy: {e}")
            return {"fitness": -1, "return": -1, "sharpe": 0, "max_drawdown": 1, "trades": 0, "equity_curve": []}

    def _calculate_summary(self, results: List[WalkForwardResult], combined_equity: List[float]) -> WalkForwardSummary:
        """Calculate summary statistics from all walk-forward results."""
        if not results:
            return WalkForwardSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, [])

        # Basic statistics
        total_windows = len(results)
        avg_train_fitness = np.mean([r.train_fitness for r in results])
        avg_test_fitness = np.mean([r.test_fitness for r in results])
        avg_overfitting_ratio = np.mean([r.overfitting_ratio for r in results])

        # Performance metrics
        total_return = np.prod([1 + r.test_return for r in results]) - 1
        avg_sharpe = np.mean([r.test_sharpe for r in results])
        max_drawdown = max([r.test_max_drawdown for r in results])

        # Win rate (percentage of profitable windows)
        profitable_windows = sum(1 for r in results if r.test_return > 0)
        win_rate = profitable_windows / total_windows

        # Stability score (consistency of performance)
        test_returns = [r.test_return for r in results]
        stability_score = 1.0 - (np.std(test_returns) / (abs(np.mean(test_returns)) + 1e-10))
        stability_score = max(0, min(1, stability_score))

        return WalkForwardSummary(
            total_windows=total_windows,
            avg_train_fitness=avg_train_fitness,
            avg_test_fitness=avg_test_fitness,
            avg_overfitting_ratio=avg_overfitting_ratio,
            total_return=total_return,
            total_sharpe=avg_sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            stability_score=stability_score,
            results=results,
        )
