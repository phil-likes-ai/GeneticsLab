# ga_lab/multi_objective.py
"""
Multi-objective optimization using NSGA-II algorithm for the genetic algorithm trading system.
Optimizes multiple objectives simultaneously: return, risk, drawdown, etc.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from .strategy import GeneticStrategy
from .manager import GeneticAlgorithmManager


@dataclass
class ObjectiveWeights:
    """Weights for different objectives in multi-objective optimization."""

    return_weight: float = 1.0
    sharpe_weight: float = 1.0
    drawdown_weight: float = 1.0
    stability_weight: float = 0.5
    trade_frequency_weight: float = 0.3


class MultiObjectiveOptimizer:
    """NSGA-II based multi-objective optimizer for trading strategies."""

    def __init__(self, cfg: Dict, logger: logging.Logger):
        self.cfg = cfg
        self.log = logger
        self.base_manager = GeneticAlgorithmManager(cfg, logger)

        # Multi-objective configuration
        mo_cfg = cfg.get("multi_objective", {})
        self.objectives = mo_cfg.get("objectives", ["return", "sharpe", "drawdown"])
        self.pareto_size = mo_cfg.get("pareto_size", 50)
        self.crowding_distance_weight = mo_cfg.get("crowding_distance_weight", 0.1)

    def optimize_pareto_front(
        self,
        db: "Database",
        symbol: str,
        timeframe: str,
        limit: Optional[int] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[GeneticStrategy]:
        """
        Run multi-objective optimization to find Pareto front.

        Returns:
            List of strategies on the Pareto front
        """
        # Load data
        candles = db.load_candles(symbol=symbol, timeframe=timeframe, limit=limit, start_ts=start_ts, end_ts=end_ts)

        if candles.empty:
            self.log.warning("No data available for multi-objective optimization")
            return []

        # Initialize population
        population = self._initialize_population()

        # Evaluate initial population
        self._evaluate_population_multi_objective(population, candles)

        # NSGA-II evolution
        for generation in range(self.base_manager.gens):
            self.log.info(f"Multi-objective generation {generation + 1}/{self.base_manager.gens}")

            # Create offspring
            offspring = self._create_offspring(population)

            # Evaluate offspring
            self._evaluate_population_multi_objective(offspring, candles)

            # Combine parent and offspring
            combined_population = population + offspring

            # Non-dominated sorting and selection
            population = self._nsga_ii_selection(combined_population)

            # Log progress
            pareto_front = self._get_pareto_front(population)
            self.log.info(f"Pareto front size: {len(pareto_front)}")

        # Return final Pareto front
        final_pareto = self._get_pareto_front(population)
        self.log.info(f"Final Pareto front contains {len(final_pareto)} strategies")

        return final_pareto

    def _initialize_population(self) -> List[GeneticStrategy]:
        """Initialize population for multi-objective optimization."""
        population = []
        for _ in range(self.base_manager.pop_size):
            strategy = self.base_manager._create_random()
            population.append(strategy)
        return population

    def _evaluate_population_multi_objective(self, population: List[GeneticStrategy], candles: pd.DataFrame):
        """Evaluate population with multiple objectives."""
        for strategy in population:
            # Run simulation
            balance = self.base_manager._simulate(strategy, candles)

            # Calculate multiple objectives
            objectives = self._calculate_objectives(strategy, balance)
            strategy.objectives = objectives

    def _calculate_objectives(self, strategy: GeneticStrategy, balance: float) -> Dict[str, float]:
        """Calculate multiple objectives for a strategy."""
        initial_balance = float(self.cfg["simulation"]["initial_balance"])

        objectives = {}

        # Objective 1: Return (maximize)
        net_return = (balance - initial_balance) / initial_balance
        objectives["return"] = net_return

        # Objective 2: Sharpe Ratio (maximize)
        sharpe = getattr(strategy, "sharpe_ratio", 0)
        objectives["sharpe"] = sharpe

        # Objective 3: Maximum Drawdown (minimize - so we negate it)
        max_dd = getattr(strategy, "max_drawdown", 1.0)
        objectives["drawdown"] = -max_dd  # Negative because we want to minimize

        # Objective 4: Stability (maximize)
        if hasattr(strategy, "daily_returns") and len(strategy.daily_returns) > 1:
            returns = np.array(strategy.daily_returns)
            stability = 1.0 - (np.std(returns) / (abs(np.mean(returns)) + 1e-10))
            objectives["stability"] = max(0, min(1, stability))
        else:
            objectives["stability"] = 0

        # Objective 5: Trade Frequency (balance - not too few, not too many)
        ideal_trades = 50  # Ideal number of trades
        trade_score = 1.0 - abs(strategy.trades - ideal_trades) / ideal_trades
        objectives["trade_frequency"] = max(0, trade_score)

        # Objective 6: Profit Factor (maximize, but capped)
        pf_score = min(strategy.profit_factor / 2.0, 1.0) if strategy.profit_factor > 0 else 0
        objectives["profit_factor"] = pf_score

        return objectives

    def _create_offspring(self, population: List[GeneticStrategy]) -> List[GeneticStrategy]:
        """Create offspring using tournament selection and crossover."""
        offspring = []

        while len(offspring) < self.base_manager.pop_size:
            # Tournament selection
            parent1 = self._tournament_selection_multi_objective(population)
            parent2 = self._tournament_selection_multi_objective(population)

            # Crossover
            if np.random.random() < self.base_manager.cross_rate:
                child = self.base_manager._crossover(parent1, parent2)
            else:
                child = self._copy_strategy(parent1)

            # Mutation
            self.base_manager._mutate(child)

            offspring.append(child)

        return offspring

    def _tournament_selection_multi_objective(self, population: List[GeneticStrategy]) -> GeneticStrategy:
        """Tournament selection for multi-objective optimization."""
        tournament_size = min(self.base_manager.k, len(population))
        tournament = np.random.choice(population, tournament_size, replace=False)

        # Select based on dominance and crowding distance
        best = tournament[0]
        for candidate in tournament[1:]:
            if self._dominates(candidate, best):
                best = candidate
            elif not self._dominates(best, candidate):
                # If neither dominates, choose based on crowding distance
                if getattr(candidate, "crowding_distance", 0) > getattr(best, "crowding_distance", 0):
                    best = candidate

        return best

    def _dominates(self, strategy1: GeneticStrategy, strategy2: GeneticStrategy) -> bool:
        """Check if strategy1 dominates strategy2 (Pareto dominance)."""
        if not hasattr(strategy1, "objectives") or not hasattr(strategy2, "objectives"):
            return False

        obj1 = strategy1.objectives
        obj2 = strategy2.objectives

        # Check if strategy1 is at least as good in all objectives
        at_least_as_good = True
        strictly_better = False

        for obj_name in self.objectives:
            if obj_name in obj1 and obj_name in obj2:
                if obj1[obj_name] < obj2[obj_name]:
                    at_least_as_good = False
                    break
                elif obj1[obj_name] > obj2[obj_name]:
                    strictly_better = True

        return at_least_as_good and strictly_better

    def _nsga_ii_selection(self, population: List[GeneticStrategy]) -> List[GeneticStrategy]:
        """NSGA-II selection: non-dominated sorting + crowding distance."""
        # Non-dominated sorting
        fronts = self._non_dominated_sort(population)

        # Select strategies for next generation
        next_population = []
        front_index = 0

        while len(next_population) + len(fronts[front_index]) <= self.base_manager.pop_size:
            # Calculate crowding distance for current front
            self._calculate_crowding_distance(fronts[front_index])
            next_population.extend(fronts[front_index])
            front_index += 1

            if front_index >= len(fronts):
                break

        # Fill remaining slots from the next front based on crowding distance
        if len(next_population) < self.base_manager.pop_size and front_index < len(fronts):
            remaining_slots = self.base_manager.pop_size - len(next_population)
            self._calculate_crowding_distance(fronts[front_index])

            # Sort by crowding distance (descending)
            fronts[front_index].sort(key=lambda s: getattr(s, "crowding_distance", 0), reverse=True)
            next_population.extend(fronts[front_index][:remaining_slots])

        return next_population

    def _non_dominated_sort(self, population: List[GeneticStrategy]) -> List[List[GeneticStrategy]]:
        """Non-dominated sorting for NSGA-II."""
        fronts = [[]]

        for strategy in population:
            strategy.domination_count = 0
            strategy.dominated_strategies = []

            for other in population:
                if self._dominates(strategy, other):
                    strategy.dominated_strategies.append(other)
                elif self._dominates(other, strategy):
                    strategy.domination_count += 1

            if strategy.domination_count == 0:
                strategy.rank = 0
                fronts[0].append(strategy)

        front_index = 0
        while len(fronts[front_index]) > 0:
            next_front = []

            for strategy in fronts[front_index]:
                for dominated in strategy.dominated_strategies:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.rank = front_index + 1
                        next_front.append(dominated)

            if next_front:
                fronts.append(next_front)
            front_index += 1

        return fronts

    def _calculate_crowding_distance(self, front: List[GeneticStrategy]):
        """Calculate crowding distance for strategies in a front."""
        if len(front) <= 2:
            for strategy in front:
                strategy.crowding_distance = float("inf")
            return

        # Initialize crowding distance
        for strategy in front:
            strategy.crowding_distance = 0

        # Calculate for each objective
        for obj_name in self.objectives:
            # Sort by objective value
            front.sort(key=lambda s: s.objectives.get(obj_name, 0))

            # Set boundary points to infinity
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            # Calculate crowding distance for intermediate points
            obj_range = front[-1].objectives.get(obj_name, 0) - front[0].objectives.get(obj_name, 0)

            if obj_range > 0:
                for i in range(1, len(front) - 1):
                    distance = (
                        front[i + 1].objectives.get(obj_name, 0) - front[i - 1].objectives.get(obj_name, 0)
                    ) / obj_range
                    front[i].crowding_distance += distance

    def _get_pareto_front(self, population: List[GeneticStrategy]) -> List[GeneticStrategy]:
        """Extract the Pareto front from population."""
        fronts = self._non_dominated_sort(population)
        return fronts[0] if fronts else []

    def _copy_strategy(self, strategy: GeneticStrategy) -> GeneticStrategy:
        """Create a copy of a strategy."""
        return GeneticStrategy(
            id=self.base_manager._new_id(),
            indicator_params=strategy.indicator_params.copy(),
            thresholds=strategy.thresholds.copy(),
            weights=strategy.weights.copy(),
        )

    def analyze_pareto_front(self, pareto_strategies: List[GeneticStrategy]) -> Dict:
        """Analyze the Pareto front and provide insights."""
        if not pareto_strategies:
            return {}

        analysis = {
            "total_strategies": len(pareto_strategies),
            "objective_ranges": {},
            "trade_offs": {},
            "recommended_strategies": {},
        }

        # Analyze objective ranges
        for obj_name in self.objectives:
            values = [s.objectives.get(obj_name, 0) for s in pareto_strategies if hasattr(s, "objectives")]
            if values:
                analysis["objective_ranges"][obj_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

        # Find recommended strategies for different risk profiles
        if len(pareto_strategies) >= 3:
            # Conservative: Low drawdown, moderate return
            conservative = min(
                pareto_strategies, key=lambda s: s.objectives.get("drawdown", 0)
            )  # Remember drawdown is negative

            # Aggressive: High return, higher risk acceptable
            aggressive = max(pareto_strategies, key=lambda s: s.objectives.get("return", 0))

            # Balanced: Best Sharpe ratio
            balanced = max(pareto_strategies, key=lambda s: s.objectives.get("sharpe", 0))

            analysis["recommended_strategies"] = {
                "conservative": conservative.id,
                "aggressive": aggressive.id,
                "balanced": balanced.id,
            }

        return analysis
