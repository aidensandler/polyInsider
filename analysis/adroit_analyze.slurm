#!/bin/bash
# Analyse the downloaded data on an Adroit compute node (no internet needed).
# Submit from ~/pm473 on Adroit:   sbatch adroit_analyze.slurm
# Before submitting: copy the code, data/live/trades_raw and data_events_full_markets.parquet to Adroit,
# and create the Python environment once (see handoff.md section 14).
# Check Princeton Research Computing's current guidance for partitions, memory limits and module names.
#SBATCH --job-name=polyinsider-analyze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$HOME/pm473"
module purge
module load anaconda3/2024.6 2>/dev/null || module load anaconda3        # module name can differ; run `module avail anaconda`
source activate pm473_env 2>/dev/null || conda activate pm473_env        # created once: conda create -n pm473_env python=3.11 ; pip install -r requirements.txt

# 1. clean, compute PnL and per-wallet tables for ALL markets (reads the trade files already in the cache)
python run_milestone1.py --out data_events_full --trades-dir data/live/trades_raw \
    --market-list data_events_full_markets.parquet --n-markets 100000 \
    --end-min 2025-01-01 --end-max 2026-11-15 --audit-onchain 0 --stage analyze

# 2. event scores and flags on the full history (writes results_v3/)
python run_flagging_v2.py --data data_events_full/live --placebo data_placebo/live --out results_v3 \
    --split-date 2026-06-01 --skip-synthetic

echo "done; copy data_events_full/live and results_v3 back to the laptop"
