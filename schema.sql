-- schema.sql – GA-Lab SQLite schema

-- 1. Historic OHLCV data
CREATE TABLE IF NOT EXISTS candles (
    ts INTEGER NOT NULL,           -- unix timestamp (ms or s)
    symbol TEXT NOT NULL,          -- e.g. BTCUSDT
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (ts, symbol)
);

-- 2. Evolved GA strategies
CREATE TABLE IF NOT EXISTS ga_strategies (
    id TEXT PRIMARY KEY,           -- e.g. 9e1d7d2b
    generation INTEGER NOT NULL,   -- evolution generation
    fitness REAL NOT NULL,         -- final balance or score
    params TEXT NOT NULL,          -- JSON of indicator params
    thresholds TEXT NOT NULL,      -- JSON of buy/sell/sl/tp thresholds
    weights TEXT NOT NULL,         -- JSON of indicator weights
    win_rate REAL NOT NULL,
    profit_factor REAL NOT NULL,
    trades INTEGER NOT NULL
);
