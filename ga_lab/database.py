# ga_lab/database.py
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .strategy import GeneticStrategy

# Canonical DDL (kept for fresh databases)
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
CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe_ts
    ON candles(symbol, timeframe, ts);
"""


class Database:
    """SQLite helper handling candles + GA strategies."""

    def __init__(self, db_path: str | Path = "ga_lab.db") -> None:
        # Use detect_types for better type handling if needed; keep default isolation level
        self._conn = sqlite3.connect(str(db_path))
        with self._conn:
            # Ensure base objects exist (no-op if already present)
            self._conn.executescript(_DDL)
        # After ensuring tables exist, run automatic migrations for legacy schemas
        self._run_automatic_migrations()

    # ------------------------------ migrations ---------------------- #
    def _table_info(self, table: str) -> list[tuple]:
        cur = self._conn.execute(f'PRAGMA table_info("{table}")')
        return cur.fetchall()

    def _has_column(self, table: str, column: str) -> bool:
        return any(row[1] == column for row in self._table_info(table))

    def _primary_key_columns(self, table: str) -> list[str]:
        # PRAGMA table_info returns rows: (cid, name, type, notnull, dflt_value, pk)
        return [row[1] for row in self._table_info(table) if row[5]]

    def _run_automatic_migrations(self) -> None:
        """
        Detects and upgrades legacy DBs:
        - Adds missing 'timeframe' column to candles
        - Rebuilds primary key to (ts, symbol, timeframe) if different
        - Adds helper index for performance
        Safe to run repeatedly (idempotent).
        """
        # 1) Ensure timeframe column exists
        needs_timeframe = not self._has_column("candles", "timeframe")
        if needs_timeframe:
            # Legacy schema had PK(ts, symbol); add timeframe with default placeholder
            with self._conn:
                self._conn.execute("ALTER TABLE candles ADD COLUMN timeframe TEXT")
                # Backfill a sensible default for legacy rows if timeframe is NULL
                # Users can later correct per-row timeframe if needed.
                self._conn.execute('UPDATE candles SET timeframe = COALESCE(timeframe, "1h") WHERE timeframe IS NULL')

        # 2) Ensure primary key matches desired composite (ts, symbol, timeframe)
        pk_cols = self._primary_key_columns("candles")
        desired_pk = ["ts", "symbol", "timeframe"]
        if pk_cols != desired_pk:
            # SQLite cannot alter primary keys; rebuild table.
            with self._conn:
                self._conn.execute("PRAGMA foreign_keys=OFF;")
                # Create a new table with the correct schema
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS candles_new(
                        ts        INTEGER,
                        symbol    TEXT,
                        timeframe TEXT,
                        open      REAL,
                        high      REAL,
                        low       REAL,
                        close     REAL,
                        volume    REAL,
                        PRIMARY KEY (ts, symbol, timeframe)
                    )
                """
                )
                # Copy distinct rows; if duplicates exist, last one wins due to PK on insert
                # Prefer explicit column list to avoid surprises.
                common_cols = ["ts", "symbol", "timeframe", "open", "high", "low", "close", "volume"]
                cols_csv = ",".join(common_cols)
                self._conn.execute(
                    f"""
                    INSERT OR REPLACE INTO candles_new({cols_csv})
                    SELECT {cols_csv} FROM candles
                """
                )
                # Replace old table
                self._conn.execute("DROP TABLE candles")
                self._conn.execute("ALTER TABLE candles_new RENAME TO candles")
                self._conn.execute("PRAGMA foreign_keys=ON;")

        # 3) Ensure performance index exists
        with self._conn:
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe_ts
                ON candles(symbol, timeframe, ts)
            """
            )

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

        # Validate and clamp LIMIT before appending to SQL.
        # Note: sqlite+pandas (read_sql_query) does not support parameterizing LIMIT,
        # so we pre-validate and enforce integer bounds to avoid SQL injection.
        safe_limit: int | None = None
        if limit is not None:
            try:
                lim_val = int(limit)
                if lim_val > 0:
                    # Clamp to a conservative upper bound
                    safe_limit = min(lim_val, 1_000_000)
            except (TypeError, ValueError):
                safe_limit = None

        if safe_limit is not None:
            query += f" LIMIT {safe_limit}"

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
        required_cols = ["ts", "symbol", "timeframe", "open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame is missing required columns. Got {df.columns}")

        with self._conn:
            df.to_sql("candles", self._conn, if_exists="append", index=False)

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
        cur = self._conn.execute("SELECT * FROM ga_strategies ORDER BY fitness DESC LIMIT ?", (limit,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    # Backward-compatible alias (scheduled for removal)
    def load_strategy_by__id(self, strategy_id: str) -> GeneticStrategy | None:
        return self.load_strategy_by_id(strategy_id)

    def load_strategy_by_id(self, strategy_id: str) -> GeneticStrategy | None:
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
