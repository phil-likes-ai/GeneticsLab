# ga_lab/strategy.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import pandas as pd


@dataclass(slots=True)
class GeneticStrategy:
    id: str
    indicator_params: Dict[str, int | float]
    thresholds: Dict[str, float]
    weights: Dict[str, float]
    fitness: float = 0.0
    generation: int = 0
    trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # ---------------- evaluation ---------------- #
    def evaluate(self, ind: pd.Series) -> Tuple[str, float]:
        buy_score: float = 0.0
        sell_score: float = 0.0
        total_weight: float = 0.0

        w = self.weights
        th = self.thresholds

        # RSI
        rsi_len = self.indicator_params["rsi_length"]
        rsi_val = ind.get(f"RSI_{rsi_len}", math.nan)
        if not math.isnan(rsi_val):
            total_weight += w["rsi"]
            ovs, ovb = th["rsi_oversold"], th["rsi_overbought"]
            if rsi_val < ovs:
                buy_score += w["rsi"] * (1 - rsi_val / ovs)
            elif rsi_val > ovb:
                sell_score += w["rsi"] * (rsi_val - ovb) / (100 - ovb)

        # MACD
        fast, slow, sig = (
            self.indicator_params[k] for k in ("macd_fast", "macd_slow", "macd_signal")
        )
        macd = ind.get(f"MACD_{fast}_{slow}_{sig}")
        macds = ind.get(f"MACDS_{fast}_{slow}_{sig}")
        if macd is not None and macds is not None:
            total_weight += w["macd"]
            if macd > macds:
                buy_score += w["macd"]
            elif macd < macds:
                sell_score += w["macd"]

        # EMA cross
        ema_s = ind.get(f"EMA_{self.indicator_params['ema_short_len']}")
        ema_l = ind.get(f"EMA_{self.indicator_params['ema_long_len']}")
        if ema_s is not None and ema_l is not None:
            total_weight += w["ema_cross"]
            if ema_s > ema_l:
                buy_score += w["ema_cross"]
            elif ema_s < ema_l:
                sell_score += w["ema_cross"]

        if total_weight:
            buy_score /= total_weight
            sell_score /= total_weight

        action: str = "hold"
        confidence: float = 0.0
        if buy_score > sell_score and buy_score > th["buy_strength"]:
            action, confidence = "buy", buy_score
        elif sell_score > buy_score and sell_score > th["sell_strength"]:
            action, confidence = "sell", sell_score
        return action, confidence
