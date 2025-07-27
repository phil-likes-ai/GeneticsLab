# ga_lab/database.py
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .strategy import GeneticStrategy

_DDL: str = """
CREATE TABLE IF NOT EXISTS candles(
    ts        INTEGER,
    symbol    TEXT,
    timeframe TEXT,
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    volume    REAL,
    PRIMARY KEY (ts, symbol, timeframe)
);
CREATE TABLE IF NOT EXISTS ga_strategies(
    id            TEXT PRIMARY KEY,
    generation    INTEGER,
    fitness       REAL,
    params        TEXT,
    thresholds    TEXT,
    weights       TEXT,
    win_rate      REAL,
    profit_factor REAL,
    trades        INTEGER
);
"""


class Database:
    """SQLite helper handling candles + GA strategies."""

    def __init__(self, db_path: str | Path = "ga_lab.db") -> None:
        self._conn = sqlite3.connect(str(db_path))
        with self._conn:
            self._conn.executescript(_DDL)

    # ------------------------------ candles ------------------------- #
    def load_candles(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        limit: int | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> pd.DataFrame:
        query = "SELECT ts, symbol, timeframe, open, high, low, close, volume FROM candles"
        params: list[int | str] = []
        where: list[str] = []

        if symbol:
            where.append("symbol = ?")
            params.append(symbol)
        if timeframe:
            where.append("timeframe = ?")
            params.append(timeframe)
        if start_ts:
            where.append("ts >= ?")
            params.append(start_ts)
        if end_ts:
            where.append("ts <= ?")
            params.append(end_ts)

        if where:
            query += " WHERE " + " AND ".join(where)

        query += " ORDER BY ts ASC"

        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql_query(query, self._conn, params=params)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
        return df

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> int | None:
        """Gets the most recent timestamp (as int) for a given symbol and timeframe."""
        query = "SELECT MAX(ts) FROM candles WHERE symbol = ? AND timeframe = ?"
        cur = self._conn.execute(query, (symbol, timeframe))
        result = cur.fetchone()[0]
        return result if result else None

    def save_candles(self, df: pd.DataFrame) -> None:
        """Saves a DataFrame of candles to the database."""
        required_cols = ['ts', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame is missing required columns. Got {df.columns}")

        with self._conn:
            df.to_sql(
                "candles", self._conn, if_exists="append", index=False
            )

    # ---------------------------- strategies ------------------------ #
    def get_all_strategies(self, limit: int | None = None) -> pd.DataFrame:
        query = "SELECT id, generation, fitness, win_rate, profit_factor, trades, params, thresholds, weights FROM ga_strategies ORDER BY fitness DESC"
        if limit:
            query += f" LIMIT {limit}"
        return pd.read_sql_query(query, self._conn)

    def save_strategies(self, strategies: Iterable[GeneticStrategy]) -> None:
        with self._conn:
            for s in strategies:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO ga_strategies
                    (id,generation,fitness,params,thresholds,weights,
                     win_rate,profit_factor,trades)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        s.id,
                        s.generation,
                        s.fitness,
                        json.dumps(s.indicator_params),
                        json.dumps(s.thresholds),
                        json.dumps(s.weights),
                        s.win_rate,
                        s.profit_factor,
                        s.trades,
                    ),
                )

    def load_top_strategies(self, limit: int = 20) -> List[dict]:
        cur = self._conn.execute(
            "SELECT * FROM ga_strategies ORDER BY fitness DESC LIMIT ?", (limit,)
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def load_strategy_by__id(self, strategy_id: str) -> GeneticStrategy | None:
        """Loads a single strategy from the database by its ID."""
        query = "SELECT id, params, thresholds, weights FROM ga_strategies WHERE id = ?"
        cur = self._conn.execute(query, (strategy_id,))
        row = cur.fetchone()
        if not row:
            return None

        id, params_json, thresholds_json, weights_json = row
        params = json.loads(params_json)
        thresholds = json.loads(thresholds_json)
        weights = json.loads(weights_json)

        return GeneticStrategy(id, params, thresholds, weights)
