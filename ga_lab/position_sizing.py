# ga_lab/position_sizing.py
"""
Advanced position sizing methods for the genetic algorithm trading system.
Implements Kelly criterion, risk parity, and other sophisticated position sizing techniques.
"""
from __future__ import annotations

import math
from typing import List, Optional
import numpy as np
import pandas as pd


class PositionSizer:
    """Advanced position sizing calculator with multiple methods."""

    def __init__(self, method: str = "fixed_fraction", **kwargs):
        self.method = method
        self.config = kwargs

    def calculate_position_size(
        self,
        balance: float,
        price: float,
        volatility: float,
        win_rate: float = 0.5,
        avg_win: float = 0.0,
        avg_loss: float = 0.0,
        returns_history: Optional[List[float]] = None,
        confidence: float = 1.0,
    ) -> float:
        """
        Calculate position size based on the configured method.

        Args:
            balance: Current account balance
            price: Current asset price
            volatility: Asset volatility (ATR or similar)
            win_rate: Historical win rate
            avg_win: Average winning trade
            avg_loss: Average losing trade
            returns_history: Historical returns for Kelly calculation
            confidence: Signal confidence (0-1)

        Returns:
            Position size as fraction of balance
        """
        if self.method == "fixed_fraction":
            return self._fixed_fraction(balance, price)
        elif self.method == "kelly":
            return self._kelly_criterion(balance, price, win_rate, avg_win, avg_loss, returns_history)
        elif self.method == "risk_parity":
            return self._risk_parity(balance, price, volatility)
        elif self.method == "volatility_adjusted":
            return self._volatility_adjusted(balance, price, volatility, confidence)
        elif self.method == "adaptive":
            return self._adaptive_sizing(balance, price, volatility, win_rate, confidence)
        else:
            return self._fixed_fraction(balance, price)

    def _fixed_fraction(self, balance: float, price: float) -> float:
        """Simple fixed fraction of balance."""
        fraction = self.config.get("risk_per_trade", 0.02)
        max_position = self.config.get("max_position_size", 1.0)
        return min(fraction, max_position)

    def _kelly_criterion(
        self,
        balance: float,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        returns_history: Optional[List[float]] = None,
    ) -> float:
        """Kelly criterion position sizing."""
        kelly_fraction = self.config.get("kelly_fraction", 0.25)  # Conservative Kelly
        max_position = self.config.get("max_position_size", 1.0)

        if returns_history and len(returns_history) > 10:
            # Use historical returns for Kelly calculation
            returns = np.array(returns_history)
            wins = returns[returns > 0]
            losses = returns[returns < 0]

            if len(wins) > 0 and len(losses) > 0:
                win_rate_calc = len(wins) / len(returns)
                avg_win_calc = np.mean(wins)
                avg_loss_calc = abs(np.mean(losses))

                if avg_loss_calc > 0:
                    # Kelly formula: f = (bp - q) / b
                    # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
                    b = avg_win_calc / avg_loss_calc
                    p = win_rate_calc
                    q = 1 - p

                    kelly_f = (b * p - q) / b
                    kelly_f = max(0, min(kelly_f, 1.0))  # Clamp to [0, 1]

                    # Apply conservative fraction
                    optimal_fraction = kelly_f * kelly_fraction
                    return min(optimal_fraction, max_position)

        # Fallback to simple Kelly if no history
        if avg_loss > 0 and win_rate > 0:
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p

            kelly_f = (b * p - q) / b
            kelly_f = max(0, min(kelly_f, 1.0))

            optimal_fraction = kelly_f * kelly_fraction
            return min(optimal_fraction, max_position)

        # Default to fixed fraction if Kelly can't be calculated
        return self._fixed_fraction(balance, price)

    def _risk_parity(self, balance: float, price: float, volatility: float) -> float:
        """Risk parity position sizing based on volatility."""
        target_volatility = self.config.get("target_volatility", 0.15)  # 15% annual vol
        max_position = self.config.get("max_position_size", 1.0)

        if volatility > 0:
            # Annualized volatility (assuming daily data)
            annual_vol = volatility * np.sqrt(252)

            # Position size inversely proportional to volatility
            vol_adjusted_fraction = target_volatility / annual_vol
            vol_adjusted_fraction = max(0.01, min(vol_adjusted_fraction, max_position))

            return vol_adjusted_fraction

        return self._fixed_fraction(balance, price)

    def _volatility_adjusted(self, balance: float, price: float, volatility: float, confidence: float) -> float:
        """Volatility-adjusted position sizing with confidence scaling."""
        base_fraction = self.config.get("risk_per_trade", 0.02)
        max_position = self.config.get("max_position_size", 1.0)
        vol_adjustment = self.config.get("volatility_adjustment", 2.0)

        if volatility > 0:
            # Reduce position size in high volatility
            vol_factor = 1.0 / (1.0 + volatility * vol_adjustment)

            # Scale by confidence
            confidence_factor = confidence**0.5  # Square root scaling

            adjusted_fraction = base_fraction * vol_factor * confidence_factor
            return min(adjusted_fraction, max_position)

        return base_fraction * confidence

    def _adaptive_sizing(
        self, balance: float, price: float, volatility: float, win_rate: float, confidence: float
    ) -> float:
        """Adaptive position sizing combining multiple factors."""
        base_fraction = self.config.get("risk_per_trade", 0.02)
        max_position = self.config.get("max_position_size", 1.0)

        # Volatility adjustment
        vol_factor = 1.0 / (1.0 + volatility * 2.0) if volatility > 0 else 1.0

        # Win rate adjustment
        win_rate_factor = min(win_rate * 2.0, 1.5) if win_rate > 0.5 else max(win_rate * 2.0, 0.5)

        # Confidence adjustment
        confidence_factor = confidence**0.5

        # Market regime adjustment (could be enhanced with regime detection)
        regime_factor = 1.0  # Placeholder for future regime detection

        # Combine all factors
        adaptive_fraction = base_fraction * vol_factor * win_rate_factor * confidence_factor * regime_factor

        return min(adaptive_fraction, max_position)


class RiskManager:
    """Risk management utilities for position sizing and trade management."""

    @staticmethod
    def calculate_stop_loss_distance(atr: float, multiplier: float = 2.0) -> float:
        """Calculate stop loss distance based on ATR."""
        return atr * multiplier

    @staticmethod
    def calculate_position_risk(
        position_size: float, entry_price: float, stop_loss_price: float, balance: float
    ) -> float:
        """Calculate the risk percentage of a position."""
        if stop_loss_price <= 0 or entry_price <= 0:
            return 0.0

        risk_per_share = abs(entry_price - stop_loss_price)
        total_risk = position_size * risk_per_share

        return total_risk / balance if balance > 0 else 0.0

    @staticmethod
    def adjust_position_for_correlation(
        base_position: float, correlation: float, existing_exposure: float = 0.0
    ) -> float:
        """Adjust position size based on correlation with existing positions."""
        if abs(correlation) > 0.7:  # High correlation
            # Reduce position size
            correlation_factor = 1.0 - abs(correlation) * 0.5
            adjusted_position = base_position * correlation_factor

            # Further reduce if we already have exposure
            if existing_exposure > 0:
                exposure_factor = 1.0 - min(existing_exposure, 0.5)
                adjusted_position *= exposure_factor

            return adjusted_position

        return base_position

    @staticmethod
    def calculate_heat(open_positions: List[dict], balance: float) -> float:
        """Calculate total portfolio heat (risk)."""
        total_risk = 0.0

        for position in open_positions:
            entry_price = position.get("entry_price", 0)
            stop_loss = position.get("stop_loss", 0)
            size = position.get("size", 0)

            if entry_price > 0 and stop_loss > 0:
                risk = abs(entry_price - stop_loss) * size
                total_risk += risk

        return total_risk / balance if balance > 0 else 0.0
