-- schema.sql – GA-Lab SQLite schema
-- Canonical schema aligned with runtime [ga_lab/database.py](ga_lab/database.py:13)

-- 1. Historic OHLCV data
CREATE TABLE IF NOT EXISTS candles (
    ts INTEGER NOT NULL,            -- unix timestamp (ms)
    symbol TEXT NOT NULL,           -- e.g. BTC/USDT
    timeframe TEXT NOT NULL,        -- e.g. 1m, 5m, 1h
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (ts, symbol, timeframe)
);

-- 2. Evolved GA strategies
CREATE TABLE IF NOT EXISTS ga_strategies (
    id TEXT PRIMARY KEY,            -- e.g. 9e1d7d2b
    generation INTEGER NOT NULL,    -- evolution generation
    fitness REAL NOT NULL,          -- final balance or score
    params TEXT NOT NULL,           -- JSON of indicator params
    thresholds TEXT NOT NULL,       -- JSON of buy/sell/sl/tp thresholds
    weights TEXT NOT NULL,          -- JSON of indicator weights
    win_rate REAL NOT NULL,
    profit_factor REAL NOT NULL,
    trades INTEGER NOT NULL
);

-- Optional helper indexes for performance (safe if they already exist)
CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe_ts
    ON candles(symbol, timeframe, ts);

-- Migration helper notes:
-- If your existing 'candles' table lacks 'timeframe':
--   1) CREATE TABLE candles_new AS SELECT ts, symbol, '1h' AS timeframe, open, high, low, close, volume FROM candles;
--   2) DROP TABLE candles;
--   3) CREATE TABLE ... (use the definition above)
--   4) INSERT INTO candles SELECT * FROM candles_new;
--   5) DROP TABLE candles_new;
-- Replace '1h' with the actual timeframe as appropriate per dataset.
