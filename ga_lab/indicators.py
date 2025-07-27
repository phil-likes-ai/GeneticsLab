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


# ---------------------------------------------------------------------- #
# Convenience wrapper that appends *all* columns required by GA engine   #
# ---------------------------------------------------------------------- #
class Indicators:
    """Vectorised indicator calculator producing canonical column names."""

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # RSI
        out["RSI_14"] = rsi(out["close"], 14)

        # MACD 12-26-9
        macd_line, macd_signal, macd_hist = macd(out["close"], 12, 26, 9)
        out["MACD_12_26_9"] = macd_line
        out["MACDS_12_26_9"] = macd_signal
        out["MACDH_12_26_9"] = macd_hist

        # Bollinger Bands 20/2
        bb_u, bb_m, bb_l = bollinger_bands(out["close"], 20, 2.0)
        out["BBU_20_2.0"] = bb_u
        out["BBM_20_2.0"] = bb_m
        out["BBL_20_2.0"] = bb_l

        # EMA short / long (8, 21)
        out["EMA_8"] = ema(out["close"], 8)
        out["EMA_21"] = ema(out["close"], 21)

        # Stochastic 14-3-3
        k_val, d_val = stoch(out["close"], out["high"], out["low"], 14, 3, 3)
        out["STOCHK_14_3_3"] = k_val
        out["STOCHD_14_3_3"] = d_val

        # ATR 14
        out["ATR_14"] = atr(out["high"], out["low"], out["close"], 14)

        return out
