# setup.sh
#!/usr/bin/env bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "GA laboratory ready. Run:"
echo "python3 -m ga_lab.cli --db ga_lab.db --symbol BTCUSDT --limit 50000"
