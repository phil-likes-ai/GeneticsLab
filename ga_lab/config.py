# ga_lab/config.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, TypeVar, Dict

_T = TypeVar("_T")


class Config(dict):  # type: ignore[misc]  # dict subclass acceptable for mapping
    """Lightweight JSON-file backed configuration."""

    def __init__(self, path: str | Path = "config.json") -> None:
        super().__init__()
        self.load(path)

    def load(self, path: str | Path) -> None:
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        self.clear()
        self.update(data)


def require(cfg: Mapping[str, Any], key: str) -> _T:  # noqa: ANN401
    """Deep-lookup helper; raises KeyError if key missing."""
    cur: Any = cfg
    for part in key.split("."):
        if part not in cur:
            raise KeyError(f"Missing key: {key}")
        cur = cur[part]
    return cur  # type: ignore[return-value]


def validate_config(cfg: Dict) -> None:
    """Validate configuration completeness and consistency."""

    # Required sections
    required_sections = ["simulation", "genetic_algorithm"]
    for section in required_sections:
        if section not in cfg:
            raise ValueError(f"Missing required config section: {section}")

    # GA parameters validation
    ga_params = cfg["genetic_algorithm"]["parameters"]
    required_ga_params = [
        "population_size",
        "mutation_rate",
        "crossover_rate",
        "elite_size",
        "generations_to_evolve",
        "tournament_k",
    ]

    for param in required_ga_params:
        if param not in ga_params:
            raise ValueError(f"Missing GA parameter: {param}")

        # Range validation
        if param == "population_size" and ga_params[param] < 10:
            raise ValueError("Population size must be >= 10")
        if param in ["mutation_rate", "crossover_rate"] and not 0 < ga_params[param] <= 1:
            raise ValueError(f"{param} must be between 0 and 1")
        if param == "elite_size" and ga_params[param] >= ga_params["population_size"]:
            raise ValueError("Elite size must be less than population size")

    # Validate parameter ranges exist
    required_ranges = ["param_ranges", "threshold_ranges", "weight_ranges"]
    for range_type in required_ranges:
        if range_type not in cfg["genetic_algorithm"]:
            raise ValueError(f"Missing {range_type} in genetic_algorithm config")

    # Validate simulation parameters
    if "initial_balance" not in cfg["simulation"]:
        raise ValueError("Missing initial_balance in simulation config")
    if cfg["simulation"]["initial_balance"] <= 0:
        raise ValueError("Initial balance must be positive")

    print("✅ Configuration validation passed")
