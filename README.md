# PolyInsiders: do Polymarket wallets that beat prices keep beating them?

ECE 473 project by Aiden Sandler and Emilly Antero (November 2026).

We study whether Polymarket trader accounts ("wallets") that outperform market prices on resolved events keep doing so afterwards, when they trade, and whether following them would have paid. We never claim private information: the evidence can only show persistent outperformance, which could also be skill, speed or patience.

**Website:** after the steps in "Publish the website" below, it is served from the `web/` folder.

## What is in this repository

| Folder | What it is |
|---|---|
| `web/` | The PolyInsiders website (static HTML, JavaScript and JSON): overview, tracker, findings, methodology, empirical results, trade timing. |
| `polyinsider/` | Python package behind the website: market and wallet models, live tracker, data import and download tools. App notes are in `README_APP.md`. |
| `data/run*/`, `data/walk_forward*/` | Saved model files, predictions and evaluation outputs used by the website. |
| `docs/`, `slides/` | Research reports, demo script and presentation decks. |
| `results_v3/`, `results_v4/` | Small result files the website scripts read (flagged-wallet lists, validation reports). Full set is in `analysis/`. |
| `GITHUB_SETUP.md`, `README_APP.md`, `START_HERE.txt` | The app's own setup notes. |
| `analysis/` | Our analysis pipeline: wallet flagging test, alert and copy-trading tests, trade timing, walkthrough notebook, results report (PDF) and tests. See `analysis/README.md` and `analysis/handoff.md`. |

Raw trade data (about 35 million trades) and the private API key are **not** in the repository.

## Main results (details in `analysis/polyinsider_results_report.pdf`)

- **Flagging wallets.** On 3,713 markets (Jan 2025 to Nov 2026) the original skill test flagged only 23 of 17,254 experienced wallets because it assumed more luck-driven spread than the data has. Measuring the spread from the data flags 207 wallets that still beat comparable wallets afterwards; a simulation with no skilled traders flags none. The flagged list is a list of likely-informed *candidates*, not proven insiders.
- **Alerts.** A trained alert model lost money out of sample (about -21% per dollar over five separate quarters). A simple rule with no model, copying wallets whose past score is at least 3 over 10 or more events, returned about +8% per dollar (95% interval about +2% to +15%) against about -4% for random trades, and about +8% (interval about -1% to +18%) on a fresh window that random trades also gained about +4% in. Modest, not statistically established, and it shrinks with trading costs.
- **Timing.** Flagged wallets trade earlier than others and in bursts in the 10 minutes before sharp price jumps (about three times the usual share, roughly 70% of those dollars on the side the price then moved to). They show no edge after a jump or in the final hours before resolution. This could reflect price impact rather than information.
- **Market-level prediction (the website's tracker).** A pre-resolution model that adds wallet-skill flow did not beat the raw market price in walk-forward tests (see `docs/REAL_RESULTS_AND_LAST_STEP.md`).

## Run the website locally

```bash
python3 -m http.server 8000 --directory web
# open http://127.0.0.1:8000
```

The app itself (tracker, models, tests):

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install duckdb      # only needed for the timing-page builders
python -m pytest -q
python -m polyinsider.cli serve --port 8000
```

## Publish the website (GitHub Pages)

A workflow in `.github/workflows/pages.yml` publishes the `web/` folder. After pushing: **Settings > Pages > Build and deployment > Source: GitHub Actions**. The site appears at `https://<user>.github.io/<repo>/` after the first run (Actions tab). The website shows saved JSON snapshots; it does not poll Polymarket by itself.

## Run the analysis

```bash
cd analysis
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install duckdb      # only needed for the timing-page builders
python -m pytest -q tests          # 54 tests
```

The full pipeline needs the downloaded trade data; the commands are in `analysis/README.md` and `analysis/handoff.md` (sections 14 to 21). The notebook `analysis/polyinsider_walkthrough.ipynb` runs in a simulated mode without the data (numbers from simulated mode are not results).

## Limits

Historical replays ignore trade-size limits and most trading costs. For about 92% of markets the time the deciding information became public is approximated by the scheduled end date. Wallet addresses are public blockchain identifiers; we label behaviour, not people. No financial advice.
