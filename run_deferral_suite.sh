#!/usr/bin/env bash
# Runs every deferral experiment in one go (about 45-60 minutes on a free
# Colab T4 GPU), then builds the figures and the combined table.
#
#   bash run_deferral_suite.sh            # full suite
#   N=200 bash run_deferral_suite.sh      # quicker, smaller samples
set -euo pipefail
N="${N:-400}"
ATTACKS="${ATTACKS:-char,word,genetic}"
run() { echo; echo ">>> deferral_study.py $*"; python deferral_study.py --n_samples "$N" --attacks "$ATTACKS" "$@"; }

# 1. Does the result hold for every classifier, and on harder benign traffic?
for primary in toxic_bert toxic_comment_distilbert roberta_hate_dynabench; do
  for benign in jigsaw jigsaw_hard; do
    run --primary "$primary" --benign_source "$benign"
  done
done

# 2. How stable is it across data samples? (two more seeds for the main setting)
for seed in 43 44; do
  run --primary toxic_bert --benign_source jigsaw --seed "$seed"
done

# 3. Benign traffic from a different platform
run --primary toxic_bert --benign_source hatexplain

# 4. Ensemble signal (rerun after the calibration fix)
run --signal ensemble

python deferral_report.py
zip -qr deferral_results.zip results/
echo; echo "Done. Results: results/deferral_summary_all.md, figures in results/figures/, all in deferral_results.zip"
