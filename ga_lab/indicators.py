# ga_lab/indicators.py
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------- #
# Native indicator implementations (NumPy-backed, MyPy-safe)             #
# ---------------------------------------------------------------------- #
def ema(series: pd.Series, length: int) -> pd.Series:  # noqa: D401
    """Exponential moving average (span = length)."""
    return series.ewm(span=length, adjust=False).mean()


def hull_ma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average (approximation using WMA via EMA)."""
    length = max(2, int(length))
    half = max(1, length // 2)
    sqrt_len = max(1, int(np.sqrt(length)))
    wma1 = ema(series, half)
    wma2 = ema(series, length)
    diff = 2 * wma1 - wma2
    return ema(diff, sqrt_len)


def rsi(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain, index=close.index).rolling(length).mean()
    avg_loss = pd.Series(loss, index=close.index).rolling(length).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series,
    fast: int,
    slow: int,
    signal_len: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_len, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    close: pd.Series,
    length: int,
    stddev: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    basis = close.rolling(length).mean()
    std = close.rolling(length).std()
    upper = basis + stddev * std
    lower = basis - stddev * std
    return upper, basis, lower


def stoch(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    k: int,
    d: int,
    smooth_k: int,
) -> tuple[pd.Series, pd.Series]:
    lowest = low.rolling(k).min()
    highest = high.rolling(k).max()
    percent_k = 100 * (close - lowest) / (highest - lowest + 1e-10)
    smoothed_k = percent_k.rolling(smooth_k).mean()
    percent_d = smoothed_k.rolling(d).mean()
    return smoothed_k, percent_d


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int,
) -> pd.Series:
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(length).mean()


# -------- Additional indicators -------- #
def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tp = (high + low + close) / 3.0
    sma = tp.rolling(length).mean()
    mad = (tp - sma).abs().rolling(length).mean() + 1e-10
    return (tp - sma) / (0.015 * mad)


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    hh = high.rolling(length).max()
    ll = low.rolling(length).min()
    return -100 * (hh - close) / (hh - ll + 1e-10)


def keltner_channels(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int, mult: float
) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = ema(close, length)
    rng = atr(high, low, close, length)
    upper = mid + mult * rng
    lower = mid - mult * rng
    return upper, mid, lower


def donchian_channels(high: pd.Series, low: pd.Series, length: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    upper = high.rolling(length).max()
    lower = low.rolling(length).min()
    mid = (upper + lower) / 2.0
    return upper, mid, lower


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


# -------- Advanced Volume-Based Indicators -------- #
def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3.0
    volume_price = typical_price * volume
    return volume_price.rolling(length).sum() / volume.rolling(length).sum()


def accumulation_distribution(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Accumulation/Distribution Line"""
    clv = ((close - low) - (high - close)) / (high - low + 1e-10)
    return (clv * volume).cumsum()


def volume_profile(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """Volume Profile - simplified as volume-weighted price momentum"""
    vwap_val = (close * volume).rolling(length).sum() / volume.rolling(length).sum()
    return (close / vwap_val - 1) * 100


# -------- Momentum Indicators -------- #
def roc(close: pd.Series, length: int) -> pd.Series:
    """Rate of Change"""
    return ((close / close.shift(length)) - 1) * 100


def cmo(close: pd.Series, length: int) -> pd.Series:
    """Chande Momentum Oscillator"""
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    sum_gain = pd.Series(gain, index=close.index).rolling(length).sum()
    sum_loss = pd.Series(loss, index=close.index).rolling(length).sum()

    return 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss + 1e-10)


def trix(close: pd.Series, length: int) -> pd.Series:
    """TRIX - Triple Exponential Average"""
    ema1 = ema(close, length)
    ema2 = ema(ema1, length)
    ema3 = ema(ema2, length)
    return (ema3 / ema3.shift(1) - 1) * 10000


# -------- Volatility Indicators -------- #
def chaikin_volatility(high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    """Chaikin Volatility"""
    hl_spread = high - low
    ema_spread = ema(hl_spread, length)
    return ((ema_spread - ema_spread.shift(length)) / ema_spread.shift(length) * 100).fillna(0)


def historical_volatility(close: pd.Series, length: int) -> pd.Series:
    """Historical Volatility (annualized)"""
    returns = np.log(close / close.shift(1))
    return returns.rolling(length).std() * np.sqrt(252) * 100


# -------- Regime Detection Indicators -------- #
def trend_strength(close: pd.Series, length: int) -> pd.Series:
    """Trend Strength using linear regression slope"""

    def calculate_slope(series):
        if len(series) < 2:
            return 0.0
        x = np.arange(len(series))
        y = series.values
        if np.std(x) == 0:
            return 0.0
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

    return close.rolling(length).apply(calculate_slope, raw=False)


def volatility_regime(close: pd.Series, length: int) -> pd.Series:
    """Volatility Regime - normalized volatility measure"""
    returns = close.pct_change()
    vol = returns.rolling(length).std()
    vol_ma = vol.rolling(length * 2).mean()
    return vol / (vol_ma + 1e-10)


def market_state(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """Market State combining price and volume momentum"""
    price_momentum = roc(close, length // 2)
    volume_momentum = roc(volume, length // 2)

    # Normalize both to [-1, 1] range
    price_norm = np.tanh(price_momentum / 100)
    volume_norm = np.tanh(volume_momentum / 100)

    return (price_norm + volume_norm) / 2


# ---------------------------------------------------------------------- #
# Convenience wrapper that appends *all* columns required by GA engine   #
# ---------------------------------------------------------------------- #
class Indicators:
    """Vectorised indicator calculator. Dynamically computes indicator columns required by the GA."""

    @staticmethod
    def add_all(df: pd.DataFrame, required_params: dict[str, int | float] | None = None) -> pd.DataFrame:
        """
        Add indicator columns to the dataframe.

        If required_params is provided (e.g., from a GeneticStrategy.indicator_params),
        compute exactly the columns referenced by that strategy.

        If not provided, fall back to a reasonable default set to keep CLI/analysis working.
        """
        out = df.copy()

        # Always ensure basic price columns exist
        close = out["close"]
        high = out["high"]
        low = out["low"]
        volume = out["volume"] if "volume" in out.columns else pd.Series(0.0, index=out.index)

        # Defaults
        default = {
            "rsi_length": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "ema_short_len": 8,
            "ema_long_len": 21,
            # New defaults
            "cci_length": 20,
            "wr_length": 14,
            "kc_length": 20,
            "kc_mult": 2.0,
            "donchian_length": 20,
            "hull_length": 14,
        }

        # Merge params
        params = (
            default
            if required_params is None
            else {
                **default,
                **{
                    k: (float(v) if isinstance(v, float) else int(v))
                    for k, v in required_params.items()
                    if isinstance(v, (int, float))
                },
            }
        )

        # RSI
        rsi_len = int(params["rsi_length"])
        out[f"RSI_{rsi_len}"] = rsi(close, rsi_len)

        # MACD
        fast = int(params["macd_fast"])
        slow = int(params["macd_slow"])
        sig = int(params["macd_signal"])
        macd_line, macd_signal, macd_hist = macd(close, fast, slow, sig)
        out[f"MACD_{fast}_{slow}_{sig}"] = macd_line
        out[f"MACDS_{fast}_{slow}_{sig}"] = macd_signal
        out[f"MACDH_{fast}_{slow}_{sig}"] = macd_hist

        # EMAs
        ema_s_len = int(params["ema_short_len"])
        ema_l_len = int(params["ema_long_len"])
        out[f"EMA_{ema_s_len}"] = ema(close, ema_s_len)
        out[f"EMA_{ema_l_len}"] = ema(close, ema_l_len)

        # Stochastic (fixed baseline)
        k_val, d_val = stoch(close, high, low, 14, 3, 3)
        out["STOCHK_14_3_3"] = k_val
        out["STOCHD_14_3_3"] = d_val

        # ATR baseline
        out["ATR_14"] = atr(high, low, close, 14)

        # -------- New Indicator Columns -------- #
        # CCI
        cci_len = int(params["cci_length"])
        out[f"CCI_{cci_len}"] = cci(high, low, close, cci_len)

        # Williams %R
        wr_len = int(params["wr_length"])
        out[f"WR_{wr_len}"] = williams_r(high, low, close, wr_len)

        # Keltner Channels
        kc_len = int(params["kc_length"])
        kc_mult = float(params["kc_mult"])
        kc_u, kc_m, kc_l = keltner_channels(high, low, close, kc_len, kc_mult)
        out[f"KC_U_{kc_len}_{kc_mult}"] = kc_u
        out[f"KC_M_{kc_len}_{kc_mult}"] = kc_m
        out[f"KC_L_{kc_len}_{kc_mult}"] = kc_l

        # Donchian Channels
        d_len = int(params["donchian_length"])
        d_u, d_m, d_l = donchian_channels(high, low, d_len)
        out[f"DON_U_{d_len}"] = d_u
        out[f"DON_M_{d_len}"] = d_m
        out[f"DON_L_{d_len}"] = d_l

        # Hull MA
        h_len = int(params["hull_length"])
        out[f"HULL_{h_len}"] = hull_ma(close, h_len)

        # OBV
        out["OBV"] = obv(close, volume)

        # -------- Advanced Volume-Based Indicators -------- #
        # VWAP
        vwap_len = int(params.get("vwap_length", 20))
        out[f"VWAP_{vwap_len}"] = vwap(high, low, close, volume, vwap_len)

        # Accumulation/Distribution
        out["AD"] = accumulation_distribution(high, low, close, volume)

        # Volume Profile
        vp_len = int(params.get("volume_profile_length", 20))
        out[f"VP_{vp_len}"] = volume_profile(close, volume, vp_len)

        # -------- Momentum Indicators -------- #
        # Rate of Change
        roc_len = int(params.get("roc_length", 14))
        out[f"ROC_{roc_len}"] = roc(close, roc_len)

        # Chande Momentum Oscillator
        cmo_len = int(params.get("cmo_length", 14))
        out[f"CMO_{cmo_len}"] = cmo(close, cmo_len)

        # TRIX
        trix_len = int(params.get("trix_length", 14))
        out[f"TRIX_{trix_len}"] = trix(close, trix_len)

        # -------- Volatility Indicators -------- #
        # Chaikin Volatility
        cv_len = int(params.get("chaikin_vol_length", 14))
        out[f"CV_{cv_len}"] = chaikin_volatility(high, low, cv_len)

        # Historical Volatility
        hv_len = int(params.get("hist_vol_length", 20))
        out[f"HV_{hv_len}"] = historical_volatility(close, hv_len)

        # -------- Regime Detection Indicators -------- #
        # Trend Strength
        ts_len = int(params.get("trend_strength_length", 20))
        out[f"TS_{ts_len}"] = trend_strength(close, ts_len)

        # Volatility Regime
        vr_len = int(params.get("vol_regime_length", 50))
        out[f"VR_{vr_len}"] = volatility_regime(close, vr_len)

        # Market State
        ms_len = int(params.get("market_state_length", 20))
        out[f"MS_{ms_len}"] = market_state(close, volume, ms_len)

        return out
