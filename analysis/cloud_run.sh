#!/bin/bash
# Run the whole data extension on a rented Linux VM (Ubuntu 22.04 or newer) and leave only results to pull back.
# NOT TESTED on a real VM (written without cloud access): read it first and run it step by step.
#
# VM suggestion: 16 vCPU and 128 GB memory (for example AWS r6i.4xlarge or GCP n2-highmem-16), 250 GB SSD,
# on-demand (not spot) for the analysis step. Stop and delete the VM and its disk when finished.
#
# 1) from your laptop, copy the CODE zip (no data, no .env) and unzip it:
#      scp pm473_code_only.zip ubuntu@VM_IP:~/ ; ssh ubuntu@VM_IP
#      unzip -q pm473_code_only.zip && cd pm473
# 2) run this script inside tmux so it survives a dropped connection:
#      tmux new -s run ; bash cloud_run.sh        (detach: Ctrl-b then d; re-attach: tmux attach -t run)
# 3) when it prints DONE, from your laptop:
#      scp ubuntu@VM_IP:~/pm473/pm473_results_for_laptop.tar.gz ~/pm473/
#      then unpack it into ~/pm473 with:  tar xzf pm473_results_for_laptop.tar.gz   (adds new folders only)
#
# No API key is needed (the on-chain audit is off). Never copy .env to a cloud machine.
set -euo pipefail
cd "$(dirname "$0")"
sudo apt-get update -y && sudo apt-get install -y python3-venv python3-pip tmux unzip
python3 -m venv ~/pm473_env && source ~/pm473_env/bin/activate
pip install -q -r requirements.txt

TAGS="fed fed-rates macro-fed federal-reserve elections global-elections world-elections primary-elections primary-election special-elections mayoral-elections economy economic-policy us-economy inflation cpi jobs-report unemployment gdp"
mkdir -p data/live/trades_raw

# The 1,610 markets already on your laptop are NOT re-downloaded here; they are covered by the existing run.
# To make a single analysis over everything, this VM downloads ALL windows into its own cache (about 15 million
# extra trades for 2026 H1-Q3 are fetched again). If you would rather save that time, copy your laptop's
# data/live/trades_raw and data_events/live/markets.parquet to the VM first and the existing files are reused.
python run_milestone1.py --out data_events_fresh --trades-dir data/live/trades_raw --end-min 2026-10-01 --end-max 2026-11-15 --n-markets 3000 --stage download --tags $TAGS
python run_milestone1.py --out data_events_2025  --trades-dir data/live/trades_raw --end-min 2025-01-01 --end-max 2025-12-31 --n-markets 6000 --stage download --tags $TAGS
python run_milestone1.py --out data_events       --trades-dir data/live/trades_raw --end-min 2026-01-01 --end-max 2026-09-30 --n-markets 2000 --stage download --tags $TAGS
# re-run any of the three lines above if it reports failed markets (downloads resume)

python combine_market_lists.py data_events/live data_events_2025/live data_events_fresh/live --out data_events_full_markets.parquet
python run_milestone1.py --out data_events_full --trades-dir data/live/trades_raw --market-list data_events_full_markets.parquet \
    --n-markets 100000 --end-min 2025-01-01 --end-max 2026-11-15 --audit-onchain 0 --stage analyze
python run_flagging_v2.py --data data_events_full/live --placebo data_placebo/live --out results_v3 --split-date 2026-06-01 --skip-synthetic || \
    echo "note: run_flagging_v2 needs data_placebo/live; copy it from the laptop or skip this step and run it on the laptop"

# results to pull back: cleaned data (needed by the notebook and site), the market list, and results_v3.
# New folder names only, so nothing on your laptop is overwritten.
tar czf pm473_results_for_laptop.tar.gz data_events_full data_events_fresh/live/markets.parquet data_events_2025/live/markets.parquet data_events_full_markets.parquet $(ls -d results_v3 2>/dev/null)
ls -la pm473_results_for_laptop.tar.gz
echo DONE
