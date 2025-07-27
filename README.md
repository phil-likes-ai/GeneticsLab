# GA-Lab – A Standalone Genetic Algorithm Laboratory for Trading Strategies

GA-Lab is a powerful, standalone tool for discovering, analyzing, and backtesting indicator-based trading strategies using a genetic algorithm. It provides a complete end-to-end workflow from data acquisition to strategy analysis.

---

## Table of Contents

1.  [Features](#features)
2.  [Setup and Installation](#setup-and-installation)
3.  [End-to-End Workflow](#end-to-end-workflow)
4.  [CLI Command Reference](#cli-command-reference)
5.  [Configuration Files](#configuration-files)
6.  [Development](#development)

---

## Features

*   **Genetic Algorithm Core**: Employs a multi-generation GA with tournament selection, single-point crossover, mutation, and elitism to evolve optimal strategy parameters.
*   **Flexible Data Pipeline**: Ingests historical candle data from any `ccxt`-supported exchange using a simple configuration file.
*   **Robust Backtesting**: Features a realistic, timeframe-aware backtesting engine to validate strategy performance on out-of-sample data.
*   **Insightful Reporting**: Generates detailed, polished HTML analysis reports with multiple visualizations (heatmaps, 3D scatter plots, etc.) to provide deep insights into strategy behavior.
*   **SQLite Backend**: Uses a simple and portable SQLite database for storing candle data and evolved strategies.
*   **High-Quality Codebase**: Enforces strict type-safety (`mypy --strict`) and clean code standards.

---

## Setup and Installation

Follow these steps to set up your local environment.

```bash
# 1. Clone the repository
git clone https://github.com/your-org/ga-lab.git
cd ga-lab

# 2. Create a Python virtual environment
# This isolates project dependencies and avoids conflicts.
python3 -m venv venv

# 3. Activate the virtual environment
# You must do this every time you open a new terminal.
source venv/bin/activate

# 4. Install all required packages
pip install -e ".[dev]"
```

After these steps, your environment is ready.

---

## End-to-End Workflow

Here is the recommended workflow for using GA-Lab:

### Step 1: Configure Your Data

Edit `data_config.json` to define which markets and timeframes you want to analyze. You can add as many symbols as you need.

```json
{
  "defaults": {
    "since": "2023-01-01"
  },
  "markets": [
    {
      "exchange": "binance",
      "symbol": "BTC/USDT",
      "timeframes": ["5m", "1h", "4h"]
    }
  ]
}
```

### Step 2: Fetch Market Data

Run the `fetch-data` command with the `--all` flag. This will automatically download all the data specified in your config file and save it to the database.

```bash
python3 -m ga_lab.cli fetch-data --all
```

### Step 3: Evolve Strategies

Run the genetic algorithm to discover trading strategies for a specific market.

```bash
python3 -m ga_lab.cli evolve --symbol BTC/USDT --timeframe 1h
```

### Step 4: Analyze the Results

Generate a detailed HTML report to understand the characteristics of the top-performing strategies. The `--visualize` and `--report` flags generate all plots and the final HTML file.

```bash
python3 -m ga_lab.cli analyze --limit 50 --visualize --report
```
This saves the report to `analysis_plots/strategy_report.html`.

### Step 5: Backtest a Specific Strategy

After identifying a promising strategy from the report (using its ID), run a dedicated backtest to validate its performance.

```bash
python3 -m ga_lab.cli backtest --strategy-id <id> --symbol BTC/USDT --timeframe 1h --plot
```

---

## CLI Command Reference

All commands are run via `python3 -m ga_lab.cli <command>`.

### 1. `fetch-data`

Downloads historical candle data.

*   **Usage (all markets):** `fetch-data --all`
*   **Usage (single market):** `fetch-data --symbol <symbol> --timeframe <timeframe>`

### 2. `evolve`

Runs the genetic algorithm to find optimal strategies.

*   **Usage:** `evolve --symbol <symbol> --timeframe <timeframe>`
*   **Arguments**:
    *   `--generations` (optional): Number of generations to run.
    *   `--population-size` (optional): Number of strategies per generation.

### 3. `analyze`

Generates a detailed HTML report and visualizations for the top strategies.

*   **Usage:** `analyze --limit <n> --visualize --report`
*   **Arguments**:
    *   `--limit` (optional): Number of top strategies to analyze (default: 20).
    *   `--visualize` (optional): Generate and save plot images.
    *   `--report` (optional): Generate the final HTML report.

### 4. `backtest`

Runs a full backtest for a single strategy.

*   **Usage:** `backtest --strategy-id <id> --symbol <symbol> --timeframe <timeframe>`
*   **Arguments**:
    *   `--plot` (optional): Generate and save an equity curve plot.

---

## Configuration Files

*   **`config.json`**: Defines the parameter ranges for indicators, thresholds, and weights used during evolution. It also contains the GA's meta-parameters (e.g., population size, mutation rate).
*   **`data_config.json`**: Specifies the exchanges, symbols, and timeframes for the `fetch-data --all` command.

---

## Development

### Setup

```bash
# Create venv and install all dependencies
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

### Quality Gates

| Tool       | Command         | Description        |
| ---------- | --------------- | ------------------ |
| **Black**  | `black ga_lab`  | Code formatting    |
| **Flake8** | `flake8 ga_lab` | Linting            |
| **MyPy**   | `mypy ga_lab`   | Strict type safety |

To run all checks:

```bash
black ga_lab && flake8 ga_lab && mypy ga_lab
```
