# ga_lab/manager.py
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import uuid
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from .constants import SECURE_RANDOM
from .indicators import Indicators
from .position_sizing import PositionSizer, RiskManager
from .strategy import GeneticStrategy


def _rand_val(values: Sequence[int | float]) -> int | float:
    if len(values) == 2:
        lo, hi = values
        # ensure numeric type stability
        if isinstance(lo, float) or isinstance(hi, float):
            return float(np.random.uniform(float(lo), float(hi)))
        return int(SECURE_RANDOM.randint(int(lo), int(hi)))
    return SECURE_RANDOM.choice(values)


def _clip_param(name: str, val: int | float, ranges: Dict[str, Sequence[int | float]]) -> int | float:
    rng = ranges.get(name)
    if not rng:
        return val
    if len(rng) == 2:
        lo, hi = float(rng[0]), float(rng[1])
        v = float(val)
        v = max(lo, min(hi, v))
        # cast back to int if original bounds are integers
        if all(isinstance(x, int) or float(x).is_integer() for x in rng):
            return int(round(v))
        return v
    # discrete set: choose nearest allowed
    return min(rng, key=lambda x: abs(float(x) - float(val)))


class GeneticAlgorithmManager:
    """Multi-generation GA engine with enhanced operators, constraints, adaptive mutation, parallel eval, and early stop."""

    def __init__(self, cfg: Dict, logger: logging.Logger) -> None:
        # Validate config first
        from .config import validate_config

        validate_config(cfg)

        self.cfg = cfg
        self.log = logger

        p = cfg["genetic_algorithm"]["parameters"]
        self.pop_size: int = p["population_size"]
        self.base_mut_rate: float = p["mutation_rate"]
        self.mut_rate: float = self.base_mut_rate
        self.cross_rate: float = p["crossover_rate"]
        self.elite: int = p["elite_size"]
        self.gens: int = p["generations_to_evolve"]
        self.k: int = p["tournament_k"]
        self.step_div: int = p.get("mutation_step_divisor", 10)

        ga = cfg["genetic_algorithm"]
        self.param_rng: Dict[str, Sequence[int | float]] = ga["param_ranges"]
        self.thresh_rng: Dict[str, Sequence[float]] = ga["threshold_ranges"]
        self.weight_rng: Dict[str, Sequence[float]] = ga["weight_ranges"]

        # Operator selection
        ops = ga.get("operators", {})
        self.crossover_kind: str = ops.get("crossover", "uniform")  # uniform|one_point|blend
        self.mutation_kind: str = ops.get("mutation", "gaussian")  # gaussian|step

        # Early stopping
        es = ga.get("early_stopping", {})
        self.patience: int = int(es.get("patience", 10))
        self.min_improve: float = float(es.get("min_improve", 1e-6))

        # Parallelism
        self.workers: int = int(cfg.get("parallel", {}).get("workers", max(mp.cpu_count() - 1, 1)))

        # Adaptive mutation decay
        self.mut_decay: float = float(ga.get("mutation_decay", 0.98))

        # Add validation logging
        self.log.info(
            f"GA initialized: pop={self.pop_size}, gens={self.gens}, "
            f"crossover={self.crossover_kind}, mutation={self.mutation_kind}"
        )
        self.log.info(f"Workers={self.workers}, patience={self.patience}")

    # ---------------- constraints & helpers ---------------- #
    def _enforce_constraints(self, params: Dict[str, int | float]) -> None:
        """Enforce parameter constraints and relationships."""

        # EMA ordering constraint
        if "ema_short_len" in params and "ema_long_len" in params:
            short_len = int(params["ema_short_len"])
            long_len = int(params["ema_long_len"])

            if short_len >= long_len:
                params["ema_short_len"] = max(1, long_len - 2)

        # MACD parameter relationships
        if all(k in params for k in ["macd_fast", "macd_slow", "macd_signal"]):
            fast = int(params["macd_fast"])
            slow = int(params["macd_slow"])
            if fast >= slow:
                params["macd_fast"] = max(1, slow - 1)

        # Clip all parameters to their defined ranges
        for param_name, value in params.items():
            config_key = self._get_config_key(param_name)
            params[param_name] = _clip_param(param_name, value, self.param_rng)

        # Re-check EMA ordering after clipping
        if "ema_short_len" in params and "ema_long_len" in params:
            if params["ema_short_len"] >= params["ema_long_len"]:
                params["ema_short_len"] = max(1, int(params["ema_long_len"]) - 2)

    # ---------------- population helpers ---------------- #
    def _new_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _normalize_params(self, params: Dict[str, int | float]) -> Dict[str, int | float]:
        """Normalize parameter names consistently."""
        normalized = params.copy()

        # Standard mapping for EMA parameters
        if "ema_short_lengths" in normalized:
            normalized["ema_short_len"] = normalized.pop("ema_short_lengths")
        if "ema_long_lengths" in normalized:
            normalized["ema_long_len"] = normalized.pop("ema_long_lengths")

        return normalized

    def _get_config_key(self, param_name: str) -> str:
        """Map parameter names to config keys."""
        mapping = {"ema_short_len": "ema_short_lengths", "ema_long_len": "ema_long_lengths"}
        return mapping.get(param_name, param_name)

    def _create_random(self) -> GeneticStrategy:
        params = {k: _rand_val(v) for k, v in self.param_rng.items()}
        params = self._normalize_params(params)
        self._enforce_constraints(params)

        thresholds = {k: float(_rand_val(v)) for k, v in self.thresh_rng.items()}
        weights = {k: float(_rand_val(v)) for k, v in self.weight_rng.items()}
        return GeneticStrategy(self._new_id(), params, thresholds, weights)

    def _tournament(self, pop: List[GeneticStrategy]) -> GeneticStrategy:
        return max(SECURE_RANDOM.sample(pop, self.k), key=lambda s: s.fitness)

    # ---------------- crossover operators ---------------- #
    def _crossover_uniform(self, a: GeneticStrategy, b: GeneticStrategy) -> GeneticStrategy:
        child_params = {
            k: (a.indicator_params[k] if SECURE_RANDOM.random() < 0.5 else b.indicator_params[k])
            for k in a.indicator_params
        }
        child_thresh = {k: (a.thresholds[k] if SECURE_RANDOM.random() < 0.5 else b.thresholds[k]) for k in a.thresholds}
        child_weights = {k: (a.weights[k] if SECURE_RANDOM.random() < 0.5 else b.weights[k]) for k in a.weights}
        self._enforce_constraints(child_params)
        return GeneticStrategy(self._new_id(), child_params, child_thresh, child_weights)

    def _crossover_one_point(self, a: GeneticStrategy, b: GeneticStrategy) -> GeneticStrategy:
        keys = list(a.indicator_params.keys())
        cut = SECURE_RANDOM.randint(1, max(1, len(keys) - 1))
        child_params = {}
        for i, k in enumerate(keys):
            child_params[k] = a.indicator_params[k] if i < cut else b.indicator_params[k]
        child_thresh = {k: (a.thresholds[k] if SECURE_RANDOM.random() < 0.5 else b.thresholds[k]) for k in a.thresholds}
        child_weights = {k: (a.weights[k] if SECURE_RANDOM.random() < 0.5 else b.weights[k]) for k in a.weights}
        self._enforce_constraints(child_params)
        return GeneticStrategy(self._new_id(), child_params, child_thresh, child_weights)

    def _crossover_blend(self, a: GeneticStrategy, b: GeneticStrategy, alpha: float = 0.5) -> GeneticStrategy:
        child_params: Dict[str, int | float] = {}
        for k, av in a.indicator_params.items():
            bv = b.indicator_params.get(k, av)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
                lo = min(float(av), float(bv))
                hi = max(float(av), float(bv))
                extent = hi - lo
                range_lo = lo - alpha * extent
                range_hi = hi + alpha * extent
                val = float(np.random.uniform(range_lo, range_hi))
                child_params[k] = val
            else:
                child_params[k] = av if SECURE_RANDOM.random() < 0.5 else bv
        child_thresh = {k: float((a.thresholds[k] + b.thresholds[k]) / 2.0) for k in a.thresholds}
        child_weights = {k: float((a.weights[k] + b.weights[k]) / 2.0) for k in a.weights}
        self._enforce_constraints(child_params)
        return GeneticStrategy(self._new_id(), child_params, child_thresh, child_weights)

    def _crossover(self, a: GeneticStrategy, b: GeneticStrategy) -> GeneticStrategy:
        if self.crossover_kind == "one_point":
            return self._crossover_one_point(a, b)
        if self.crossover_kind == "blend":
            return self._crossover_blend(a, b)
        return self._crossover_uniform(a, b)

    # ---------------- mutation operators ---------------- #
    def _mutate(self, s: GeneticStrategy) -> None:
        for k in list(s.indicator_params.keys()):
            if SECURE_RANDOM.random() < self.mut_rate:
                config_key = self._get_config_key(k)
                rng = self.param_rng.get(config_key)
                if not rng:
                    continue
                v = s.indicator_params[k]
                if self.mutation_kind == "gaussian" and len(rng) == 2:
                    lo, hi = float(rng[0]), float(rng[1])
                    sigma = (hi - lo) / max(1, self.step_div)
                    new_v = float(v) + float(np.random.normal(0.0, sigma))
                else:
                    new_v = _rand_val(rng)
                s.indicator_params[k] = _clip_param(config_key, new_v, self.param_rng)

        for k in list(s.thresholds.keys()):
            if SECURE_RANDOM.random() < self.mut_rate:
                s.thresholds[k] = float(_rand_val(self.thresh_rng[k]))

        for k in list(s.weights.keys()):
            if SECURE_RANDOM.random() < self.mut_rate:
                s.weights[k] = float(_rand_val(self.weight_rng[k]))

        self._enforce_constraints(s.indicator_params)  # Add this line

    # ---------------- fitness simulation ---------------- #
    def _simulate(self, strat: GeneticStrategy, df: pd.DataFrame) -> float:
        # Dynamic indicators per strategy parameters
        ind_df = Indicators.add_all(df, required_params=strat.indicator_params)

        # Initialize simulation state
        sim_cfg = self.cfg["simulation"]
        initial_balance = float(sim_cfg["initial_balance"])
        balance = initial_balance
        position_size = 0.0  # Actual position size (can be fractional)
        entry_price = 0.0

        # Trading costs
        transaction_cost = float(sim_cfg.get("transaction_cost_pct", 0.001))
        slippage = float(sim_cfg.get("slippage_pct", 0.0005))

        # Risk management
        risk_cfg = sim_cfg.get("risk_management", {})
        enable_sl = risk_cfg.get("enable_stop_loss", True)
        enable_tp = risk_cfg.get("enable_take_profit", True)
        max_dd_stop = risk_cfg.get("max_drawdown_stop", 0.20)

        # Position sizing
        pos_cfg = sim_cfg.get("position_sizing", {})
        risk_per_trade = pos_cfg.get("risk_per_trade", 0.02)
        max_position = pos_cfg.get("max_position_size", 1.0)

        # Initialize position sizer (extract method to avoid duplicate parameter)
        method = pos_cfg.get("method", "fixed_fraction")
        pos_cfg_without_method = {k: v for k, v in pos_cfg.items() if k != "method"}
        position_sizer = PositionSizer(method=method, **pos_cfg_without_method)

        # Tracking variables
        trades = wins = losses = 0
        total_wins = total_losses = 0.0
        equity_curve = [initial_balance]
        peak_equity = initial_balance
        max_drawdown = 0.0
        consecutive_losses = 0
        daily_returns = []

        # Stop loss and take profit levels
        stop_loss_price = 0.0
        take_profit_price = 0.0

        for i in range(len(ind_df)):
            current_row = ind_df.iloc[i]
            price = df.iloc[i]["close"]
            high = df.iloc[i]["high"]
            low = df.iloc[i]["low"]

            # Calculate current equity
            if position_size != 0:
                unrealized_pnl = position_size * (price - entry_price)
                current_equity = balance + unrealized_pnl
            else:
                current_equity = balance

            equity_curve.append(current_equity)

            # Update peak and drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity

            current_drawdown = (peak_equity - current_equity) / peak_equity
            max_drawdown = max(max_drawdown, current_drawdown)

            # Emergency stop if max drawdown exceeded
            if current_drawdown > max_dd_stop:
                if position_size != 0:
                    # Force close position
                    pnl = position_size * (price - entry_price)
                    balance += pnl - abs(pnl * transaction_cost)  # Apply transaction cost
                    position_size = 0.0
                break

            # Check stop loss and take profit
            if position_size != 0 and enable_sl and enable_tp:
                hit_stop = False
                exit_price = price

                if position_size > 0:  # Long position
                    if enable_sl and low <= stop_loss_price:
                        hit_stop = True
                        exit_price = stop_loss_price
                    elif enable_tp and high >= take_profit_price:
                        hit_stop = True
                        exit_price = take_profit_price
                elif position_size < 0:  # Short position
                    if enable_sl and high >= stop_loss_price:
                        hit_stop = True
                        exit_price = stop_loss_price
                    elif enable_tp and low <= take_profit_price:
                        hit_stop = True
                        exit_price = take_profit_price

                if hit_stop:
                    # Close position due to stop loss or take profit
                    pnl = position_size * (exit_price - entry_price)
                    # Apply slippage and transaction costs
                    slippage_cost = abs(position_size * exit_price * slippage)
                    transaction_cost_amount = abs(pnl * transaction_cost)
                    net_pnl = pnl - slippage_cost - transaction_cost_amount

                    balance += net_pnl

                    if net_pnl > 0:
                        wins += 1
                        total_wins += net_pnl
                        consecutive_losses = 0
                    else:
                        losses += 1
                        total_losses += abs(net_pnl)
                        consecutive_losses += 1

                    trades += 1
                    position_size = 0.0
                    continue

            # Get trading signal
            action, confidence = strat.evaluate(current_row)

            if action == "buy" and position_size <= 0:
                # Close short position if exists
                if position_size < 0:
                    pnl = position_size * (price - entry_price)
                    slippage_cost = abs(position_size * price * slippage)
                    transaction_cost_amount = abs(pnl * transaction_cost)
                    net_pnl = pnl - slippage_cost - transaction_cost_amount

                    balance += net_pnl

                    if net_pnl > 0:
                        wins += 1
                        total_wins += net_pnl
                        consecutive_losses = 0
                    else:
                        losses += 1
                        total_losses += abs(net_pnl)
                        consecutive_losses += 1

                    trades += 1

                # Open long position with advanced position sizing
                if balance > 0:
                    atr_value = current_row.get("ATR_14", price * 0.02)  # Fallback to 2% if ATR not available

                    # Calculate position size using advanced methods
                    position_fraction = position_sizer.calculate_position_size(
                        balance=balance,
                        price=price,
                        volatility=atr_value / price,  # Normalized volatility
                        win_rate=strat.win_rate,
                        avg_win=total_wins / max(wins, 1),
                        avg_loss=total_losses / max(losses, 1),
                        returns_history=strat.daily_returns[-50:] if len(strat.daily_returns) > 10 else None,
                        confidence=confidence,
                    )

                    position_size = (balance * position_fraction) / price

                    entry_price = price * (1 + slippage)  # Apply slippage

                    # Set stop loss and take profit
                    if enable_sl:
                        sl_mult = strat.thresholds.get("sl_atr_mult", 2.0)
                        stop_loss_price = entry_price - (atr_value * sl_mult)
                    if enable_tp:
                        tp_mult = strat.thresholds.get("tp_atr_mult", 3.0)
                        take_profit_price = entry_price + (atr_value * tp_mult)

            elif action == "sell" and position_size >= 0:
                # Close long position if exists
                if position_size > 0:
                    pnl = position_size * (price - entry_price)
                    slippage_cost = abs(position_size * price * slippage)
                    transaction_cost_amount = abs(pnl * transaction_cost)
                    net_pnl = pnl - slippage_cost - transaction_cost_amount

                    balance += net_pnl

                    if net_pnl > 0:
                        wins += 1
                        total_wins += net_pnl
                        consecutive_losses = 0
                    else:
                        losses += 1
                        total_losses += abs(net_pnl)
                        consecutive_losses += 1

                    trades += 1

                # Open short position with advanced position sizing
                if balance > 0:
                    atr_value = current_row.get("ATR_14", price * 0.02)

                    # Calculate position size using advanced methods
                    position_fraction = position_sizer.calculate_position_size(
                        balance=balance,
                        price=price,
                        volatility=atr_value / price,  # Normalized volatility
                        win_rate=strat.win_rate,
                        avg_win=total_wins / max(wins, 1),
                        avg_loss=total_losses / max(losses, 1),
                        returns_history=strat.daily_returns[-50:] if len(strat.daily_returns) > 10 else None,
                        confidence=confidence,
                    )

                    position_size = -(balance * position_fraction) / price  # Negative for short

                    entry_price = price * (1 - slippage)  # Apply slippage for short

                    # Set stop loss and take profit for short
                    if enable_sl:
                        sl_mult = strat.thresholds.get("sl_atr_mult", 2.0)
                        stop_loss_price = entry_price + (atr_value * sl_mult)
                    if enable_tp:
                        tp_mult = strat.thresholds.get("tp_atr_mult", 3.0)
                        take_profit_price = entry_price - (atr_value * tp_mult)

        # Close final position if exists
        if position_size != 0:
            final_price = df.iloc[-1]["close"]
            pnl = position_size * (final_price - entry_price)
            slippage_cost = abs(position_size * final_price * slippage)
            transaction_cost_amount = abs(pnl * transaction_cost)
            net_pnl = pnl - slippage_cost - transaction_cost_amount

            balance += net_pnl

            if net_pnl > 0:
                wins += 1
                total_wins += net_pnl
            else:
                losses += 1
                total_losses += abs(net_pnl)

            trades += 1

        # Store enhanced metrics in strategy
        strat.trades = trades
        strat.win_rate = wins / trades if trades else 0.0
        strat.profit_factor = (total_wins / total_losses) if total_losses > 0 else (total_wins + 1.0)

        # Store additional metrics for enhanced fitness calculation
        strat.max_drawdown = max_drawdown
        strat.equity_curve = equity_curve
        strat.consecutive_losses = consecutive_losses

        # Calculate daily returns for Sharpe ratio
        if len(equity_curve) > 1:
            returns = [(equity_curve[i] / equity_curve[i - 1] - 1) for i in range(1, len(equity_curve))]
            strat.daily_returns = returns
        else:
            strat.daily_returns = []

        # Use enhanced fitness calculation
        strat.fitness = self._calculate_fitness(balance, strat)
        return balance

    def _calculate_fitness(self, balance: float, strat: GeneticStrategy) -> float:
        """Calculate sophisticated multi-factor fitness score."""
        initial_balance = float(self.cfg["simulation"]["initial_balance"])
        net_profit = balance - initial_balance

        # Get fitness configuration
        fitness_cfg = self.cfg.get("fitness", {})
        return_weight = fitness_cfg.get("return_weight", 0.4)
        sharpe_weight = fitness_cfg.get("sharpe_weight", 0.3)
        drawdown_weight = fitness_cfg.get("drawdown_weight", 0.2)
        trade_freq_weight = fitness_cfg.get("trade_frequency_weight", 0.1)
        min_trades = fitness_cfg.get("min_trades", 10)
        max_dd_penalty = fitness_cfg.get("max_drawdown_penalty", 2.0)
        sharpe_target = fitness_cfg.get("sharpe_target", 1.0)

        if strat.trades < min_trades:
            return -1.0  # Penalty for insufficient trades

        # 1. Return Component (normalized)
        return_ratio = net_profit / initial_balance
        return_score = return_ratio

        # 2. Sharpe Ratio Component
        sharpe_ratio = self._calculate_sharpe_ratio(strat.daily_returns)
        strat.sharpe_ratio = sharpe_ratio
        # Normalize Sharpe ratio (target of 1.0 = score of 1.0)
        sharpe_score = min(sharpe_ratio / sharpe_target, 2.0) if sharpe_ratio > 0 else -0.5

        # 3. Drawdown Component (penalty)
        max_dd = strat.max_drawdown
        if max_dd > 0.5:  # Extreme drawdown
            drawdown_score = -max_dd_penalty
        elif max_dd > 0.3:  # High drawdown
            drawdown_score = -max_dd * 2.0
        elif max_dd > 0.1:  # Moderate drawdown
            drawdown_score = -max_dd
        else:  # Low drawdown (bonus)
            drawdown_score = 0.1 - max_dd

        # 4. Trade Frequency Component (normalized)
        trade_freq_score = min(strat.trades / 100.0, 1.0)

        # 5. Additional Risk Metrics
        sortino_ratio = self._calculate_sortino_ratio(strat.daily_returns)
        strat.sortino_ratio = sortino_ratio

        calmar_ratio = return_ratio / max(max_dd, 0.01)  # Avoid division by zero
        strat.calmar_ratio = calmar_ratio

        # 6. Profit Factor Bonus (capped)
        profit_factor_bonus = min(strat.profit_factor / 2.0, 0.5) if strat.profit_factor > 1.0 else 0.0

        # 7. Win Rate Bonus
        win_rate_bonus = (strat.win_rate - 0.5) * 0.2 if strat.win_rate > 0.5 else 0.0

        # 8. Consecutive Loss Penalty
        consecutive_loss_penalty = -min(strat.consecutive_losses / 10.0, 0.3)

        # Combine all components
        fitness = (
            return_score * return_weight
            + sharpe_score * sharpe_weight
            + drawdown_score * drawdown_weight
            + trade_freq_score * trade_freq_weight
            + profit_factor_bonus * 0.1
            + win_rate_bonus
            + consecutive_loss_penalty
        )

        return fitness

    def _calculate_sharpe_ratio(self, returns: list, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from daily returns."""
        if not returns or len(returns) < 2:
            return 0.0

        import numpy as np

        returns_array = np.array(returns)

        if np.std(returns_array) == 0:
            return 0.0

        excess_returns = returns_array - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized

    def _calculate_sortino_ratio(self, returns: list, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if not returns or len(returns) < 2:
            return 0.0

        import numpy as np

        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252

        # Only consider negative returns for downside deviation
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return float("inf") if np.mean(excess_returns) > 0 else 0.0

        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0

        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)  # Annualized

    # ---------------- evaluation ---------------- #
    def _eval_one(self, args: tuple[GeneticStrategy, pd.DataFrame]) -> GeneticStrategy:
        s, df = args
        self._simulate(s, df)
        return s

    def _evaluate_population(self, population: List[GeneticStrategy], candles: pd.DataFrame) -> None:
        if self.workers and self.workers > 1:
            # Use top-level compatible callable (bound method works if the class is picklable).
            # Avoid lambdas/closures which are not picklable on some platforms.
            with mp.Pool(processes=self.workers) as pool:
                population[:] = list(pool.map(self._eval_one, [(s, candles) for s in population]))
        else:
            for s in population:
                self._simulate(s, candles)

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

        best_fitness = -math.inf
        stagnation = 0

        for gen in tqdm(range(self.gens + 1), desc="Generations", position=0):
            self._evaluate_population(population, candles)

            # Calculate diversity and adapt mutation
            diversity = self._calculate_diversity(population)
            self._adaptive_mutation_rate(gen, diversity)

            for s in population:
                s.generation = gen

            population.sort(key=lambda s: s.fitness, reverse=True)
            cur_best = population[0].fitness

            # Enhanced logging
            self.log.info(
                f"Gen {gen}: best={cur_best:.4f}, diversity={diversity:.3f}, " f"mut_rate={self.mut_rate:.4f}"
            )

            # Early stopping by stagnation
            if cur_best > best_fitness + self.min_improve:
                best_fitness = cur_best
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= self.patience:
                    self.log.info("Early stopping at gen %d due to stagnation.", gen)
                    break

            elites = [
                GeneticStrategy(
                    id=e.id,
                    indicator_params=e.indicator_params.copy(),
                    thresholds=e.thresholds.copy(),
                    weights=e.weights.copy(),
                    fitness=e.fitness,
                    generation=e.generation,
                    trades=e.trades,
                    win_rate=e.win_rate,
                    profit_factor=e.profit_factor,
                    max_drawdown=e.max_drawdown,
                    sharpe_ratio=e.sharpe_ratio,
                    sortino_ratio=e.sortino_ratio,
                    calmar_ratio=e.calmar_ratio,
                    consecutive_losses=e.consecutive_losses,
                    equity_curve=e.equity_curve.copy() if e.equity_curve else [],
                    daily_returns=e.daily_returns.copy() if e.daily_returns else [],
                )
                for e in population[: self.elite]
            ]
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
                        weights=parent.weights.copy(),
                    )
                self._mutate(child)
                new_population.append(child)

            population = new_population

        self.log.info("Evolution complete. Saving final population to database...")
        db.save_strategies(population)

        return population

    def _calculate_diversity(self, population: List[GeneticStrategy]) -> float:
        """Calculate population diversity using parameter variance."""
        if len(population) < 2:
            return 0.0

        # Collect all parameter values
        param_values = {}
        for strat in population:
            for param, value in strat.indicator_params.items():
                if param not in param_values:
                    param_values[param] = []
                param_values[param].append(float(value))

        # Calculate average coefficient of variation
        diversities = []
        for param, values in param_values.items():
            if len(set(values)) > 1:  # Avoid division by zero
                mean_val = np.mean(values)
                if mean_val != 0:
                    cv = np.std(values) / abs(mean_val)
                    diversities.append(cv)

        return np.mean(diversities) if diversities else 0.0

    def _adaptive_mutation_rate(self, generation: int, diversity: float) -> None:
        """Adapt mutation rate based on diversity and generation."""
        base_rate = self.base_mut_rate

        # Increase mutation if diversity is low
        diversity_factor = 1.0 + (0.5 * (1.0 - min(diversity, 1.0)))

        # Apply generation-based decay
        decay_factor = self.mut_decay**generation

        self.mut_rate = max(0.01, base_rate * diversity_factor * decay_factor)
