# ga_lab/constants.py
from __future__ import annotations

import random
from typing import Final

SECURE_RANDOM: Final[random.SystemRandom] = random.SystemRandom()

REQUIRED_PARAMETER_KEYS: Final[list[str]] = [
    "population_size",
    "mutation_rate",
    "crossover_rate",
    "elite_size",
    "generations_to_evolve",
    "tournament_k",
    "mutation_step_divisor",
]

REQUIRED_PARAM_RANGE_KEYS: Final[list[str]] = [
    "rsi_length",
    "macd_fast",
    "macd_slow",
    "macd_signal",
    "bb_length",
    "bb_std",
    "ema_short_lengths",
    "ema_long_lengths",
    "stoch_k",
    "stoch_d",
    "stoch_smooth_k",
    "atr_length",
]

REQUIRED_THRESHOLD_RANGE_KEYS: Final[list[str]] = [
    "rsi_oversold",
    "rsi_overbought",
    "stoch_oversold",
    "stoch_overbought",
    "buy_strength",
    "sell_strength",
    "sl_atr_mult",
    "tp_atr_mult",
]

REQUIRED_WEIGHT_RANGE_KEYS: Final[list[str]] = [
    "rsi",
    "macd",
    "bollinger",
    "ema_cross",
    "stochastic",
]
