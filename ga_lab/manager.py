# ga_lab/manager.py
from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .constants import SECURE_RANDOM
from .indicators import Indicators
from .strategy import GeneticStrategy


def _rand_val(values: Sequence[int | float]) -> int | float:
    if len(values) == 2:
        lo, hi = values
        return np.random.uniform(lo, hi) if isinstance(lo, float) else SECURE_RANDOM.randint(lo, hi)
    return SECURE_RANDOM.choice(values)


class GeneticAlgorithmManager:
    """Multi-generation GA engine (native indicators)."""

    def __init__(self, cfg: Dict, logger: logging.Logger) -> None:
        self.cfg = cfg
        self.log = logger

        p = cfg["genetic_algorithm"]["parameters"]
        self.pop_size: int = p["population_size"]
        self.mut_rate: float = p["mutation_rate"]
        self.cross_rate: float = p["crossover_rate"]
        self.elite: int = p["elite_size"]
        self.gens: int = p["generations_to_evolve"]
        self.k: int = p["tournament_k"]

        ga = cfg["genetic_algorithm"]
        self.param_rng: Dict[str, Sequence[int | float]] = ga["param_ranges"]
        self.thresh_rng: Dict[str, Sequence[float]] = ga["threshold_ranges"]
        self.weight_rng: Dict[str, Sequence[float]] = ga["weight_ranges"]

    # ---------------- population helpers ---------------- #
    def _new_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _create_random(self) -> GeneticStrategy:
        params = {k: _rand_val(v) for k, v in self.param_rng.items()}
        # Map configuration parameter names to expected names
        if "ema_short_lengths" in params:
            params["ema_short_len"] = params.pop("ema_short_lengths")
        if "ema_long_lengths" in params:
            params["ema_long_len"] = params.pop("ema_long_lengths")

        thresholds = {k: float(_rand_val(v)) for k, v in self.thresh_rng.items()}
        weights = {k: float(_rand_val(v)) for k, v in self.weight_rng.items()}
        return GeneticStrategy(self._new_id(), params, thresholds, weights)

    def _tournament(self, pop: List[GeneticStrategy]) -> GeneticStrategy:
        return max(SECURE_RANDOM.sample(pop, self.k), key=lambda s: s.fitness)

    def _crossover(self, a: GeneticStrategy, b: GeneticStrategy) -> GeneticStrategy:
        child_params = {
            k: a.indicator_params[k] if SECURE_RANDOM.random() < 0.5 else b.indicator_params[k]
            for k in a.indicator_params
        }
        child_thresh = {k: a.thresholds[k] if SECURE_RANDOM.random() < 0.5 else b.thresholds[k] for k in a.thresholds}
        child_weights = {k: a.weights[k] if SECURE_RANDOM.random() < 0.5 else b.weights[k] for k in a.weights}
        return GeneticStrategy(self._new_id(), child_params, child_thresh, child_weights)

    def _mutate(self, s: GeneticStrategy) -> None:
        # Map parameter names for mutation
        param_mapping = {
            "ema_short_len": "ema_short_lengths",
            "ema_long_len": "ema_long_lengths"
        }
        
        for k in s.indicator_params:
            if SECURE_RANDOM.random() < self.mut_rate:
                config_key = param_mapping.get(k, k)
                if config_key in self.param_rng:
                    s.indicator_params[k] = _rand_val(self.param_rng[config_key])
        
        for k in s.thresholds:
            if SECURE_RANDOM.random() < self.mut_rate:
                s.thresholds[k] = float(_rand_val(self.thresh_rng[k]))
        
        for k in s.weights:
            if SECURE_RANDOM.random() < self.mut_rate:
                s.weights[k] = float(_rand_val(self.weight_rng[k]))

    # ---------------- fitness simulation ---------------- #
    def _simulate(self, strat: GeneticStrategy, df: pd.DataFrame) -> float:
        ind_df = Indicators.add_all(df)

        balance: float = 1_000.0
        entry: float = 0.0
        pos: int = 0  # 1 long, -1 short
        wins = losses = trades = 0

        for i in range(len(ind_df)):
            action, _conf = strat.evaluate(ind_df.iloc[i])
            price = df.iloc[i]["close"]

            if action == "buy" and pos <= 0:
                if pos < 0:  # close short
                    pnl = entry - price
                    balance += pnl
                    wins += pnl > 0
                    losses += pnl < 0
                    trades += 1
                entry, pos = price, 1

            elif action == "sell" and pos >= 0:
                if pos > 0:  # close long
                    pnl = price - entry
                    balance += pnl
                    wins += pnl > 0
                    losses += pnl < 0
                    trades += 1
                entry, pos = price, -1

        if pos != 0:  # close end-of-data
            pnl = (df.iloc[-1]["close"] - entry) * pos
            balance += pnl
            wins += pnl > 0
            losses += pnl < 0
            trades += 1

        strat.trades = trades
        strat.win_rate = wins / trades if trades else 0.0
        strat.profit_factor = (wins + 1) / (losses + 1)
        # --- Fitness Calculation ---
        # Penalize strategies that don't trade at all.
        if strat.trades == 0:
            strat.fitness = 0.0
            return

        net_profit = balance - 1000.0

        # Heavily reward profit factor and net profit. Penalize low trade count slightly.
        # Add a constant to ensure fitness is always positive for the tournament selection.
        fitness = (net_profit * strat.profit_factor) + (strat.trades / 10.0)

        # Ensure fitness is not negative
        strat.fitness = max(0.0, fitness)
        return balance

    # ---------------- evolution loop -------------------- #
    def evolve(
        self,
        db: "Database",
        symbol: str,
        timeframe: str,
        limit: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[GeneticStrategy]:
        candles: pd.DataFrame = db.load_candles(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            start_ts=start_ts,
            end_ts=end_ts,
        )
        if candles.empty:
            self.log.warning("No candle data found for %s %s, evolution cannot proceed.", symbol, timeframe)
            return []
        population = [self._create_random() for _ in range(self.pop_size)]

        for gen in tqdm(range(self.gens + 1), desc="Generations", position=0):
            # Progress bar for fitness evaluation within each generation
            for s in tqdm(population, desc=f"Evaluating Gen {gen}", position=1, leave=False):
                self._simulate(s, candles)
                s.generation = gen
            population.sort(key=lambda s: s.fitness, reverse=True)
            self.log.info("Gen %d best fitness %.2f", gen, population[0].fitness)

            elites = [GeneticStrategy(
                id=e.id,
                indicator_params=e.indicator_params.copy(),
                thresholds=e.thresholds.copy(),
                weights=e.weights.copy(),
                fitness=e.fitness,
                generation=e.generation,
                trades=e.trades,
                win_rate=e.win_rate,
                profit_factor=e.profit_factor
            ) for e in population[: self.elite]]
            new_population: List[GeneticStrategy] = elites.copy()

            while len(new_population) < self.pop_size:
                parent = self._tournament(population)
                if SECURE_RANDOM.random() < self.cross_rate:
                    child = self._crossover(parent, self._tournament(population))
                else:
                    child = GeneticStrategy(
                        id=self._new_id(),
                        indicator_params=parent.indicator_params.copy(),
                        thresholds=parent.thresholds.copy(),
                        weights=parent.weights.copy()
                    )
                self._mutate(child)
                new_population.append(child)

            population = new_population

        self.log.info("Evolution complete. Saving final population to database...")
        db.save_strategies(population)

        return population
