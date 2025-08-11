import time
import ccxt
import pandas as pd

DEFAULT_SLEEP_SEC = 0.2
MAX_PAGES = 50_000  # hard safety upper bound


def fetch_candles(exchange_name: str, symbol: str, timeframe: str, since_ts: int) -> pd.DataFrame:
    """Fetches historical OHLCV data from the specified exchange with rate-limit safety and loop guards."""
    exchange = getattr(ccxt, exchange_name)()

    if not exchange.has.get("fetchOHLCV", False):
        raise NotImplementedError(f"{exchange_name} does not support fetching OHLCV data.")

    all_candles = []
    limit = 1000  # Number of candles to fetch per request
    pages = 0
    last_ts = None

    # Determine polite sleep based on exchange rate limit, fallback to default
    sleep_sec = max((getattr(exchange, "rateLimit", 200) / 1000.0), DEFAULT_SLEEP_SEC)

    while True:
        pages += 1
        if pages > MAX_PAGES:
            break  # safety guard

        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)
        if not candles:
            break

        all_candles.extend(candles)
        newest_ts = candles[-1][0]

        # Break if timestamp does not advance (avoid infinite loops)
        if last_ts is not None and newest_ts <= last_ts:
            break
        last_ts = newest_ts

        since_ts = newest_ts + 1  # Move to the next candle time

        # If the exchange returns fewer candles than the limit, we've reached the end.
        if len(candles) < limit:
            break

        # Be polite to the exchange API
        time.sleep(sleep_sec)

    if not all_candles:
        return pd.DataFrame()

    # Convert to pandas DataFrame, keeping 'ts' as an integer
    df = pd.DataFrame(all_candles, columns=["ts", "open", "high", "low", "close", "volume"])

    # Add symbol and timeframe to match our database schema
    df["symbol"] = symbol
    df["timeframe"] = timeframe

    # Reorder columns to be explicit and return
    return df[["ts", "symbol", "timeframe", "open", "high", "low", "close", "volume"]]
