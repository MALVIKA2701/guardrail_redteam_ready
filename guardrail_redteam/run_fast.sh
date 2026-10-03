#!/usr/bin/env bash
# Fast end-to-end run on local surrogates (no API keys needed).
# CPU laptop: ~30-60 min.  With a GPU: ~10-15 min.
# For bigger "paper" numbers, use the full commands in README.md.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 0. offline tests + mock smoke test"
python -m pytest tests/ -q
python run_experiment.py --mock --n_samples 8

echo "== 1. main attack evaluation (toxic-bert)"
python run_experiment.py --n_samples 100 --attacks char,word,genetic

echo "== 2. stats (bootstrap CIs, Holm-corrected McNemar)"
python stats.py

echo "== 3. cross-model study + transfer matrix (3 surrogates)"
python multi_model_study.py --n_samples 60 --attacks char,word

echo "== 4. universal trigger (optimize / held-out split)"
python universal_trigger_study.py --n_samples 150 --n_seeds 3

echo "== 5. ablations"
python ablations.py --n_samples 40

echo "== 6. defense (adversarial fine-tuning)"
python defense/adversarial_finetune.py --n_eval_samples 60 --n_clean_eval_samples 100

echo "== 7. figures + report"
python visualize_results.py
python generate_report.py
echo "Done -> open results/report.md"
