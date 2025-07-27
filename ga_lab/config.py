# ga_lab/config.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, TypeVar

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
