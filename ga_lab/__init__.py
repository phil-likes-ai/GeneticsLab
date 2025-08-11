# ga_lab/__init__.py
"""
Standalone Genetic-Algorithm Laboratory (PEP 8 & MyPy-clean).
"""
from __future__ import annotations

from .config import Config
from .database import Database
from .indicators import Indicators
from .manager import GeneticAlgorithmManager
from .strategy import GeneticStrategy
from .position_sizing import PositionSizer, RiskManager
from .walk_forward import WalkForwardValidator, WalkForwardResult, WalkForwardSummary
from .multi_objective import MultiObjectiveOptimizer, ObjectiveWeights
from .enhanced_backtester import EnhancedBacktester, PerformanceMetrics, TradeRecord

__all__ = [
    "Config",
    "Database",
    "Indicators",
    "GeneticStrategy",
    "GeneticAlgorithmManager",
    "PositionSizer",
    "RiskManager",
    "WalkForwardValidator",
    "WalkForwardResult",
    "WalkForwardSummary",
    "MultiObjectiveOptimizer",
    "ObjectiveWeights",
    "EnhancedBacktester",
    "PerformanceMetrics",
    "TradeRecord",
]
