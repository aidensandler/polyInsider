# PolyInsiders — ECE 473

This repository contains the PolyInsiders prediction-market
research website, model code, trained model files, historical
evaluation results, and wallet timing analysis.

## Run locally

Create a Python virtual environment and install dependencies:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m pip install duckdb

Start the local live collector:

    python -m polyinsider.cli serve --live --interval 120 --port 8000

Open http://localhost:8000/

Alternatively, preview saved website files without live collection:

    python3 -m http.server 8000 --directory web

## Website pages

- Overview: index.html
- Alerts and copy-trading tests: alerts.html
- Live Tracker: tracker.html
- Real Results: empirical.html
- Trader Timing: timing.html
- Methodology: methodology.html

## Data and limitations

The complete raw Parquet archives and expanded model-ready
training CSV are not included because of their size.

Existing models and derived results are included. Full
retraining requires the separately maintained historical data.

Historically flagged wallets are not proven insiders.
Some public-information timestamps are approximate.

This is a local research prototype, not a permanently
hosted live service.
