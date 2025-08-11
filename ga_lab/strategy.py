# ga_lab/strategy.py
from __future__ import annotations

import math
from dataclasses import dataclass
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

    # Enhanced risk metrics
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    consecutive_losses: int = 0
    equity_curve: list = None
    daily_returns: list = None

    def __post_init__(self):
        if self.equity_curve is None:
            self.equity_curve = []
        if self.daily_returns is None:
            self.daily_returns = []

    # ---------------- evaluation ---------------- #
    def evaluate(self, ind: pd.Series) -> Tuple[str, float]:
        """
        Score buy/sell using weighted signals. Each signal contributes its weight
        to buy or sell depending on its bullish/bearish state. Scores are normalized
        by total applied weight.
        """
        buy_score: float = 0.0
        sell_score: float = 0.0
        total_weight: float = 0.0

        w = self.weights
        th = self.thresholds
        p = self.indicator_params

        def apply_weight(cond_buy: bool | float, cond_sell: bool | float, weight_key: str) -> None:
            nonlocal buy_score, sell_score, total_weight
            weight = float(w.get(weight_key, 0.0))
            if weight <= 0:
                return
            # Interpret numbers as confidence [0..1], booleans as hard 0/1
            b_conf = float(cond_buy) if isinstance(cond_buy, (int, float)) else (1.0 if cond_buy else 0.0)
            s_conf = float(cond_sell) if isinstance(cond_sell, (int, float)) else (1.0 if cond_sell else 0.0)
            if b_conf > 0 or s_conf > 0:
                total_weight += weight
                buy_score += weight * b_conf
                sell_score += weight * s_conf

        # RSI
        rsi_len = int(p.get("rsi_length", 14))
        rsi_val = ind.get(f"RSI_{rsi_len}", math.nan)
        if not math.isnan(rsi_val):
            ovs, ovb = th["rsi_oversold"], th["rsi_overbought"]
            buy_conf = max(0.0, (ovs - rsi_val) / max(ovs, 1e-9)) if rsi_val < ovs else 0.0
            sell_conf = max(0.0, (rsi_val - ovb) / max(100 - ovb, 1e-9)) if rsi_val > ovb else 0.0
            apply_weight(buy_conf, sell_conf, "rsi")

        # MACD
        fast = int(p.get("macd_fast", 12))
        slow = int(p.get("macd_slow", 26))
        sig = int(p.get("macd_signal", 9))
        macd_line = ind.get(f"MACD_{fast}_{slow}_{sig}")
        macd_signal = ind.get(f"MACDS_{fast}_{slow}_{sig}")
        if macd_line is not None and macd_signal is not None:
            apply_weight(macd_line > macd_signal, macd_line < macd_signal, "macd")

        # EMA cross
        ema_s_len = int(p.get("ema_short_len", 8))
        ema_l_len = int(p.get("ema_long_len", 21))
        ema_s = ind.get(f"EMA_{ema_s_len}")
        ema_l = ind.get(f"EMA_{ema_l_len}")
        if ema_s is not None and ema_l is not None:
            apply_weight(ema_s > ema_l, ema_s < ema_l, "ema_cross")

        # Bollinger Bands (if weighted)
        # Use proximity to bands as confidence
        bb_len = int(p.get("bb_length", 20))
        bb_std = float(p.get("bb_std", 2.0))
        bb_u = ind.get(f"BBU_{bb_len}_{bb_std}")
        bb_l = ind.get(f"BBL_{bb_len}_{bb_std}")
        close = ind.get("close", math.nan)
        if bb_u is not None and bb_l is not None and not math.isnan(close):
            width = max(bb_u - bb_l, 1e-9)
            buy_conf = max(0.0, (bb_u - close) / width) if close <= bb_l else 0.0
            sell_conf = max(0.0, (close - bb_l) / width) if close >= bb_u else 0.0
            apply_weight(buy_conf, sell_conf, "bollinger")

        # Stochastic (fixed 14-3-3)
        k_val = ind.get("STOCHK_14_3_3", math.nan)
        d_val = ind.get("STOCHD_14_3_3", math.nan)
        if not math.isnan(k_val) and not math.isnan(d_val):
            buy_conf = 1.0 if (k_val < th["stoch_oversold"] and k_val > d_val) else 0.0
            sell_conf = 1.0 if (k_val > th["stoch_overbought"] and k_val < d_val) else 0.0
            apply_weight(buy_conf, sell_conf, "stochastic")

        # ATR-based SL/TP thresholds are used at backtesting layer; here we focus on entries
        # ---------------- New Indicators ---------------- #
        # CCI
        cci_len = int(p.get("cci_length", 20))
        cci_val = ind.get(f"CCI_{cci_len}", math.nan)
        if not math.isnan(cci_val):
            buy_conf = max(0.0, (100 - cci_val) / 200.0) if cci_val < -100 else 0.0
            sell_conf = max(0.0, (cci_val - 100) / 200.0) if cci_val > 100 else 0.0
            apply_weight(buy_conf, sell_conf, "cci")

        # Williams %R ([-100, 0])
        wr_len = int(p.get("wr_length", 14))
        wr_val = ind.get(f"WR_{wr_len}", math.nan)
        if not math.isnan(wr_val):
            buy_conf = max(0.0, (-80 - wr_val) / 20.0) if wr_val < -80 else 0.0
            sell_conf = max(0.0, (wr_val - -20) / 20.0) if wr_val > -20 else 0.0
            apply_weight(buy_conf, sell_conf, "williams_r")

        # Keltner Channels
        kc_len = int(p.get("kc_length", 20))
        kc_mult = float(p.get("kc_mult", 2.0))
        kc_u = ind.get(f"KC_U_{kc_len}_{kc_mult}")
        kc_l = ind.get(f"KC_L_{kc_len}_{kc_mult}")
        if kc_u is not None and kc_l is not None and not math.isnan(close):
            width = max(kc_u - kc_l, 1e-9)
            buy_conf = max(0.0, (kc_u - close) / width) if close <= kc_l else 0.0
            sell_conf = max(0.0, (close - kc_l) / width) if close >= kc_u else 0.0
            apply_weight(buy_conf, sell_conf, "keltner")

        # Donchian Channels
        d_len = int(p.get("donchian_length", 20))
        don_u = ind.get(f"DON_U_{d_len}")
        don_l = ind.get(f"DON_L_{d_len}")
        if don_u is not None and don_l is not None and not math.isnan(close):
            mid = (don_u + don_l) / 2.0
            apply_weight(close > mid, close < mid, "donchian")

        # Hull MA trend
        h_len = int(p.get("hull_length", 14))
        hull = ind.get(f"HULL_{h_len}")
        if hull is not None and not math.isnan(close):
            apply_weight(close > hull, close < hull, "hull")

        # OBV momentum (use slope sign)
        obv_val = ind.get("OBV", math.nan)
        obv_prev = ind.get("OBV_prev", math.nan) if "OBV_prev" in ind else math.nan
        if not math.isnan(obv_val) and not math.isnan(obv_prev):
            apply_weight(obv_val > obv_prev, obv_val < obv_prev, "obv")

        # -------- Advanced Volume-Based Indicators -------- #
        # VWAP
        vwap_len = int(p.get("vwap_length", 20))
        vwap_val = ind.get(f"VWAP_{vwap_len}")
        if vwap_val is not None and not math.isnan(close):
            # Price above VWAP = bullish, below = bearish
            deviation = abs(close - vwap_val) / max(vwap_val, 1e-9)
            buy_conf = min(deviation, 1.0) if close > vwap_val else 0.0
            sell_conf = min(deviation, 1.0) if close < vwap_val else 0.0
            apply_weight(buy_conf, sell_conf, "vwap")

        # Volume Profile
        vp_len = int(p.get("volume_profile_length", 20))
        vp_val = ind.get(f"VP_{vp_len}", math.nan)
        if not math.isnan(vp_val):
            # Positive VP = bullish volume, negative = bearish
            buy_conf = max(0.0, vp_val / 10.0) if vp_val > 0 else 0.0
            sell_conf = max(0.0, -vp_val / 10.0) if vp_val < 0 else 0.0
            apply_weight(buy_conf, sell_conf, "volume_profile")

        # -------- Momentum Indicators -------- #
        # Rate of Change
        roc_len = int(p.get("roc_length", 14))
        roc_val = ind.get(f"ROC_{roc_len}", math.nan)
        if not math.isnan(roc_val):
            # Normalize ROC to [0, 1] confidence
            buy_conf = max(0.0, min(roc_val / 10.0, 1.0)) if roc_val > 0 else 0.0
            sell_conf = max(0.0, min(-roc_val / 10.0, 1.0)) if roc_val < 0 else 0.0
            apply_weight(buy_conf, sell_conf, "roc")

        # Chande Momentum Oscillator
        cmo_len = int(p.get("cmo_length", 14))
        cmo_val = ind.get(f"CMO_{cmo_len}", math.nan)
        if not math.isnan(cmo_val):
            # CMO ranges from -100 to +100
            buy_conf = max(0.0, (cmo_val + 100) / 200.0) if cmo_val > 0 else 0.0
            sell_conf = max(0.0, (100 - cmo_val) / 200.0) if cmo_val < 0 else 0.0
            apply_weight(buy_conf, sell_conf, "cmo")

        # TRIX
        trix_len = int(p.get("trix_length", 14))
        trix_val = ind.get(f"TRIX_{trix_len}", math.nan)
        if not math.isnan(trix_val):
            # TRIX crossover signal
            buy_conf = max(0.0, min(trix_val / 50.0, 1.0)) if trix_val > 0 else 0.0
            sell_conf = max(0.0, min(-trix_val / 50.0, 1.0)) if trix_val < 0 else 0.0
            apply_weight(buy_conf, sell_conf, "trix")

        # -------- Volatility-Based Signals -------- #
        # Chaikin Volatility
        cv_len = int(p.get("chaikin_vol_length", 14))
        cv_val = ind.get(f"CV_{cv_len}", math.nan)
        if not math.isnan(cv_val):
            # High volatility can signal trend continuation or reversal
            vol_threshold = th.get("volatility_threshold", 0.5)
            vol_signal = 1.0 if abs(cv_val) > vol_threshold else 0.0
            # Use with trend indicators for better signals
            apply_weight(vol_signal * 0.5, vol_signal * 0.5, "chaikin_volatility")

        # -------- Regime Detection Signals -------- #
        # Trend Strength
        ts_len = int(p.get("trend_strength_length", 20))
        ts_val = ind.get(f"TS_{ts_len}", math.nan)
        if not math.isnan(ts_val):
            trend_threshold = th.get("trend_threshold", 0.5)
            buy_conf = max(0.0, ts_val) if ts_val > trend_threshold else 0.0
            sell_conf = max(0.0, -ts_val) if ts_val < -trend_threshold else 0.0
            apply_weight(buy_conf, sell_conf, "trend_strength")

        # Volatility Regime
        vr_len = int(p.get("vol_regime_length", 50))
        vr_val = ind.get(f"VR_{vr_len}", math.nan)
        if not math.isnan(vr_val):
            # High volatility regime = more cautious signals
            vol_adjustment = 1.0 / max(vr_val, 0.5) if vr_val > 1.0 else 1.0
            # This acts as a signal dampener in high volatility
            apply_weight(vol_adjustment * 0.3, vol_adjustment * 0.3, "volatility_regime")

        # Normalize
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
