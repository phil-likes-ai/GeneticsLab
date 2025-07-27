# ga_lab/cli.py
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .analysis import view_results
from .backtester import Backtester
from .config import Config
from .data_fetcher import fetch_candles
from .database import Database
from .manager import GeneticAlgorithmManager


def _normalize_symbol(symbol: str) -> str:
    """Ensures a trading symbol is in the format BASE/QUOTE."""
    if "/" in symbol:
        return symbol.upper()
    # Simple assumption for common crypto pairs
    common_quotes = ["USDT", "BUSD", "BTC", "ETH", "USD"]
    for quote in common_quotes:
        if symbol.endswith(quote):
            base = symbol[:-len(quote)]
            return f"{base}/{quote}".upper()
    return symbol.upper()

def _setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

def _run_evolve(args: argparse.Namespace, cfg: Config, db: Database) -> None:
    """Handler for the 'evolve' command."""
    manager = GeneticAlgorithmManager(cfg, logging.getLogger())
    manager.evolve(
        db=db,
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
    )

def _run_backtest(args: argparse.Namespace, cfg: Config, db: Database) -> None:
    """Handler for the 'backtest' command."""
    print(f"Running backtest for strategy {args.strategy_id}...")

    # Load the strategy
    strategy = db.load_strategy_by_id(args.strategy_id)
    if not strategy:
        logging.error(f"Strategy with ID '{args.strategy_id}' not found.")
        return

    # Convert date strings to timestamps if they exist
    start_ts = int(pd.to_datetime(args.start_date).timestamp() * 1000) if args.start_date else None
    end_ts = int(pd.to_datetime(args.end_date).timestamp() * 1000) if args.end_date else None

    # Load candle data for the backtest period
    candles = db.load_candles(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    if candles.empty:
        logging.warning("No candle data found for the specified backtest period.")
        return

    # Run the backtest
    results = Backtester.run(
        strategy,
        candles,
        initial_balance=cfg["simulation"]["initial_balance"],
        generate_plot=args.plot
    )

    # Print the results
    print("--- Backtest Results ---")
    for key, value in results.items():
        if key != 'equity_curve':
            print(f"{key.replace('_', ' ').title()}: {value}")

    if 'plot_path' in results:
        print(f"Equity curve plot saved to: {results['plot_path']}")

def _fetch_and_save(
    db: Database, exchange: str, symbol: str, timeframe: str, since: str
) -> None:
    """Helper to fetch and save candles for a single market."""
    try:
        latest_ts = db.get_latest_timestamp(symbol, timeframe)

        start_ts: int
        if latest_ts:
            # Start fetching from the millisecond after the last candle to avoid duplicates.
            start_ts = latest_ts + 1
            print(
                f"Last candle for {symbol} {timeframe} found at "
                f"{pd.to_datetime(latest_ts, unit='ms', utc=True)}. "
                f"Fetching new data..."
            )
        else:
            # First time fetching for this market, convert the date string from the config.
            start_ts = int(pd.to_datetime(since, utc=True).timestamp() * 1000)
            print(f"No existing data for {symbol} {timeframe}. Fetching since {since}...")

        candles_df = fetch_candles(
            exchange_name=exchange,
            symbol=symbol,
            timeframe=timeframe,
            since_ts=start_ts,
        )

        if not candles_df.empty:
            print(f"Fetched {len(candles_df)} new candles. Saving to database...")
            db.save_candles(candles_df)
            print("Data saved successfully.")
        else:
            print("No new data to fetch.")
    except Exception as e:
        logging.error(f"An error for {symbol}/{timeframe}: {e}")

def _run_fetch_data(args: argparse.Namespace, db: Database) -> None:
    """Handler for the 'fetch-data' command."""
    if args.all:
        try:
            with open(args.data_config) as f:
                data_cfg = json.load(f)
        except FileNotFoundError:
            logging.error(f"Data config file not found at: {args.data_config}")
            return
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from: {args.data_config}")
            return

        default_since = data_cfg.get("defaults", {}).get("since", "2024-01-01")
        for market in data_cfg.get("markets", []):
            for timeframe in market.get("timeframes", []):
                _fetch_and_save(
                    db=db,
                    exchange=market["exchange"],
                    symbol=market["symbol"],
                    timeframe=timeframe,
                    since=market.get("since", default_since),
                )
    else:
        if not all([args.symbol, args.timeframe, args.since]):
            print("Error: --symbol, --timeframe, and --since are required unless using --all.")
            return
        _fetch_and_save(db, args.exchange, args.symbol, args.timeframe, args.since)

def _run_analyze(args: argparse.Namespace, db: Database) -> None:
    """Handler for the 'analyze' command."""
    view_results(db, top_n=args.limit, visualize=args.visualize, report=args.report)

def main() -> None:  # pragma: no cover
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Genetic Algorithm Laboratory for Trading Strategies")
    parser.add_argument("--db", default="ga_lab.db", help="Path to the SQLite database file.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Evolve command ---
    evolve_parser = subparsers.add_parser("evolve", help="Evolve a new trading strategy")
    evolve_parser.add_argument("--symbol", type=str, required=True, help="Symbol to use for evolution (e.g., BTC/USDT)")
    evolve_parser.add_argument("--timeframe", type=str, default="5m", help="Timeframe to use (e.g., 5m, 1h, 1d)")
    evolve_parser.add_argument("--limit", type=int, help="Limit the number of candles to use")
    evolve_parser.add_argument("--start-ts", type=int, help="Start timestamp (ms) for candle data")
    evolve_parser.add_argument("--end-ts", type=int, help="End timestamp (ms) for candle data")
    evolve_parser.add_argument("--config", default="config.json", help="Path to configuration file")
    evolve_parser.set_defaults(func=_run_evolve)

    # --- Analyze command ---
    analyze_parser = subparsers.add_parser("analyze", help="Analyze stored GA strategies")
    analyze_parser.add_argument("--limit", type=int, default=20, help="Number of top strategies to analyze")
    analyze_parser.add_argument("--report", action="store_true", help="Generate and save an HTML report")
    analyze_parser.add_argument("--visualize", action="store_true", help="Generate and save analysis plots")
    analyze_parser.set_defaults(func=_run_analyze)

    # --- Backtest command ---
    backtest_parser = subparsers.add_parser("backtest", help="Backtest a stored strategy")
    backtest_parser.add_argument("--strategy-id", type=str, required=True, help="ID of the strategy to backtest")
    backtest_parser.add_argument("--symbol", type=str, required=True, help="Symbol to backtest on (e.g., BTC/USDT)")
    backtest_parser.add_argument("--timeframe", type=str, required=True, help="Timeframe to use (e.g., 5m, 1h, 1d)")
    backtest_parser.add_argument("--start-date", type=str, help="Start date for backtest (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", type=str, help="End date for backtest (YYYY-MM-DD)")
    backtest_parser.add_argument("--plot", action="store_true", help="Generate an equity curve plot.")
    backtest_parser.add_argument("--config", default="config.json", help="Path to configuration file")
    backtest_parser.set_defaults(func=_run_backtest)

    # --- Fetch Data command ---
    fetch_parser = subparsers.add_parser("fetch-data", help="Fetch candle data from an exchange")
    fetch_parser.add_argument("--all", action="store_true", help="Fetch all markets from the data config file.")
    fetch_parser.add_argument("--data-config", default="data_config.json", help="Path to data configuration file")
    fetch_parser.add_argument("--exchange", type=str, help="Exchange name (e.g., binance)")
    fetch_parser.add_argument("--symbol", type=str, help="Symbol to fetch (e.g., BTC/USDT)")
    fetch_parser.add_argument("--timeframe", type=str, help="Timeframe (e.g., 1m, 5m, 1h, 1d)")
    fetch_parser.add_argument("--since", type=str, help="Start date to fetch from (YYYY-MM-DD)")
    fetch_parser.set_defaults(func=_run_fetch_data)

    args = parser.parse_args()

    # If no command is given, print help and exit
    if not hasattr(args, "func") or not args.command:
        parser.print_help()
        return

    # Normalize the symbol format for all commands that use it.
    if hasattr(args, "symbol") and args.symbol:
        args.symbol = _normalize_symbol(args.symbol)

    _setup_logging()
    db = Database(args.db)

    # Dispatch to the correct handler based on the command
    if args.command in ["evolve", "backtest"]:
        cfg = Config(Path(args.config))
        args.func(args, cfg, db)
    else:
        args.func(args, db)


if __name__ == "__main__":  # pragma: no cover
    main()
