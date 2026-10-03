"""
Generates all result visualizations as PNG files under results/figures/:
  - ASR bar chart by attack type (with bootstrap CI error bars)
  - Cross-model / API transfer heatmap
  - Query budget vs ASR curve (ablation)
  - Similarity threshold vs ASR curve (ablation)
  - Edit distance vs success scatter (perturbation cost)
  - Before/after defense comparison bar chart

Usage:
    python visualize_results.py
"""

import json
import os

import config


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise SystemExit(
            "matplotlib is required for visualize_results.py. "
            "Install it with `pip install matplotlib` (already in requirements.txt)."
        )


def plot_asr_by_attack(plt, records_path, stats_path, out_path):
    if not os.path.exists(records_path):
        print(f"  [skip] {records_path} not found")
        return
    with open(records_path) as f:
        records = [json.loads(line) for line in f]

    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f).get("bootstrap_ci_by_attack", {})

    attacks = sorted(set(r["attack"] for r in records))
    asrs, err_low, err_high = [], [], []
    for a in attacks:
        s = stats.get(a)
        if s and s["asr"] is not None:
            asrs.append(s["asr"])
            err_low.append(s["asr"] - s["ci_low"])
            err_high.append(s["ci_high"] - s["asr"])
        else:
            subset = [r for r in records if r["attack"] == a]
            asr = sum(r["success"] for r in subset) / len(subset) if subset else 0
            asrs.append(asr)
            err_low.append(0)
            err_high.append(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(attacks, asrs, yerr=[err_low, err_high], capsize=5, color="#4C72B0")
    ax.set_ylabel("Attack Success Rate")
    ax.set_title("ASR by Attack Type (95% bootstrap CI)")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_transfer_heatmap(plt, matrix_path, out_path, title):
    if not os.path.exists(matrix_path):
        print(f"  [skip] {matrix_path} not found")
        return
    with open(matrix_path) as f:
        matrix = json.load(f)

    sources = list(matrix.keys())
    targets = sorted({t for row in matrix.values() for t in row.keys()})
    data = [[matrix[s].get(t) or 0 for t in targets] for s in sources]

    fig, ax = plt.subplots(figsize=(1.5 * len(targets) + 2, 1.5 * len(sources) + 2))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right")
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(sources)
    for i in range(len(sources)):
        for j in range(len(targets)):
            ax.text(j, i, f"{data[i][j]:.2f}", ha="center", va="center", color="white")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Transfer rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_ablation_sweep(plt, ablation_path, key, xlabel, out_path):
    if not os.path.exists(ablation_path):
        print(f"  [skip] {ablation_path} not found")
        return
    with open(ablation_path) as f:
        data = json.load(f)
    sweep = data.get(key)
    if not sweep:
        return
    x_key = list(sweep[0].keys())[0]
    xs = [d[x_key] for d in sweep]
    ys = [d["asr"] for d in sweep]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o", color="#DD8452")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"ASR vs {xlabel}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_edit_distance_vs_success(plt, records_path, out_path):
    if not os.path.exists(records_path):
        print(f"  [skip] {records_path} not found")
        return
    with open(records_path) as f:
        records = [json.loads(line) for line in f]

    succ_dist = [r["edit_distance"] for r in records if r["success"]]
    fail_dist = [r["edit_distance"] for r in records if not r["success"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(fail_dist, bins=20, alpha=0.6, label="failed", color="#C44E52")
    ax.hist(succ_dist, bins=20, alpha=0.6, label="succeeded", color="#55A868")
    ax.set_xlabel("Normalized edit distance")
    ax.set_ylabel("Count")
    ax.set_title("Perturbation Cost: Success vs Failure")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_defense_comparison(plt, defense_path, out_path):
    if not os.path.exists(defense_path):
        print(f"  [skip] {defense_path} not found")
        return
    with open(defense_path) as f:
        data = json.load(f)

    before = data["before_defense"]
    after = data["after_defense"]
    attacks = sorted(set(before.keys()) & set(after.keys()) - {"overall"})
    if not attacks:
        attacks = ["overall"]

    before_vals = [before[a]["attack_success_rate"] for a in attacks if before[a].get("n", 0) > 0]
    after_vals = [after[a]["attack_success_rate"] for a in attacks if after[a].get("n", 0) > 0]
    labels = [a for a in attacks if before[a].get("n", 0) > 0]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    ax.bar([i - width / 2 for i in x], before_vals, width, label="Before defense", color="#C44E52")
    ax.bar([i + width / 2 for i in x], after_vals, width, label="After defense", color="#55A868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Effect of Adversarial Fine-tuning on ASR")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_defense_clean_data_tradeoff(plt, defense_path, out_path):
    """
    Plots clean-data accuracy and false-positive rate before vs after
    defense, side by side with the ASR drop -- so the "did the defense just
    start flagging everything" question has a visible answer, not just an
    ASR number.
    """
    if not os.path.exists(defense_path):
        print(f"  [skip] {defense_path} not found")
        return
    with open(defense_path) as f:
        data = json.load(f)
    clean = data.get("clean_data_evaluation")
    if not clean:
        print(f"  [skip] no clean_data_evaluation in {defense_path} (re-run defense/adversarial_finetune.py)")
        return

    metrics = ["accuracy", "false_positive_rate", "false_negative_rate"]
    before_vals = [clean["before_defense"][m] for m in metrics]
    after_vals = [clean["after_defense"][m] for m in metrics]

    x = range(len(metrics))
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.35
    ax.bar([i - width / 2 for i in x], before_vals, width, label="Before defense", color="#C44E52")
    ax.bar([i + width / 2 for i in x], after_vals, width, label="After defense", color="#55A868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Accuracy", "False Positive Rate", "False Negative Rate"])
    ax.set_ylim(0, 1)
    ax.set_title("Clean-Data Cost of the Defense")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    plt = _ensure_matplotlib()
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    print("Generating figures...")
    plot_asr_by_attack(
        plt, config.ATTACK_RESULTS_PATH, config.STATS_RESULTS_PATH,
        os.path.join(config.FIGURES_DIR, "asr_by_attack.png"),
    )
    plot_transfer_heatmap(
        plt, config.SURROGATE_TRANSFER_MATRIX_PATH,
        os.path.join(config.FIGURES_DIR, "cross_model_transfer.png"),
        "Cross-Model Transfer Rate",
    )
    plot_ablation_sweep(
        plt, config.ABLATION_RESULTS_PATH, "query_budget_sweep", "Query budget",
        os.path.join(config.FIGURES_DIR, "ablation_query_budget.png"),
    )
    plot_ablation_sweep(
        plt, config.ABLATION_RESULTS_PATH, "similarity_threshold_sweep", "Min. semantic similarity",
        os.path.join(config.FIGURES_DIR, "ablation_similarity_threshold.png"),
    )
    plot_edit_distance_vs_success(
        plt, config.ATTACK_RESULTS_PATH,
        os.path.join(config.FIGURES_DIR, "edit_distance_vs_success.png"),
    )
    plot_defense_comparison(
        plt, config.DEFENSE_RESULTS_PATH,
        os.path.join(config.FIGURES_DIR, "defense_before_after.png"),
    )
    plot_defense_clean_data_tradeoff(
        plt, config.DEFENSE_RESULTS_PATH,
        os.path.join(config.FIGURES_DIR, "defense_clean_data_tradeoff.png"),
    )
    print(f"Done. Figures in {config.FIGURES_DIR}/")


if __name__ == "__main__":
    main()
