import ccxt
import pandas as pd
from datetime import datetime, timezone

def fetch_candles(exchange_name: str, symbol: str, timeframe: str, since_ts: int) -> pd.DataFrame:
    """Fetches historical OHLCV data from the specified exchange."""
    exchange = getattr(ccxt, exchange_name)()

    if not exchange.has['fetchOHLCV']:
        raise NotImplementedError(f"{exchange_name} does not support fetching OHLCV data.")

    # The 'since_ts' parameter is now passed in directly.

    all_candles = []
    limit = 1000  # Number of candles to fetch per request

    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)
        if not candles:
            break

        all_candles.extend(candles)
        since_ts = candles[-1][0] + 1  # Move to the next candle time

        # If the exchange returns fewer candles than the limit, we've reached the end.
        if len(candles) < limit:
            break

    if not all_candles:
        return pd.DataFrame()

    # Convert to pandas DataFrame, keeping 'ts' as an integer
    df = pd.DataFrame(all_candles, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])

    # Add symbol and timeframe to match our database schema
    df['symbol'] = symbol
    df['timeframe'] = timeframe

    # Reorder columns to be explicit and return
    return df[['ts', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']]
