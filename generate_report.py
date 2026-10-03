"""
Aggregates every results/*.json(l) file produced by the other scripts into a
single results/report.md with embedded figure references -- meant to be the
one document you hand to a supervisor/reviewer, or paste sections from into
a portfolio writeup.

Run this LAST, after run_experiment.py, multi_model_study.py, stats.py,
ablations.py, defense/adversarial_finetune.py, and visualize_results.py.

Usage:
    python generate_report.py
"""

import json
import os

import config


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_jsonl(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    lines = []
    lines.append("# Adversarial Robustness of LLM Guardrail Models -- Report\n")
    lines.append(
        "Auto-generated summary of all experiments run in this project. "
        "See README.md for methodology and threat model details.\n"
    )

    # --- Section 1: primary attack results -----------------------------
    records = _load_jsonl(config.ATTACK_RESULTS_PATH)
    stats = _load_json(config.STATS_RESULTS_PATH)
    if records:
        lines.append("## 1. Attack Success Rate by Method\n")
        attacks = sorted(set(r["attack"] for r in records))
        lines.append("| Attack | n | ASR | 95% CI | Avg Queries | Avg Edit Dist |")
        lines.append("|---|---|---|---|---|---|")
        ci_by_attack = (stats or {}).get("bootstrap_ci_by_attack", {})
        for a in attacks:
            subset = [r for r in records if r["attack"] == a]
            n = len(subset)
            asr = sum(r["success"] for r in subset) / n if n else 0
            avg_q = sum(r["queries_used"] for r in subset) / n if n else 0
            avg_ed = sum(r["edit_distance"] for r in subset) / n if n else 0
            ci = ci_by_attack.get(a)
            ci_str = f"[{ci['ci_low']:.2%}, {ci['ci_high']:.2%}]" if ci and ci.get("ci_low") is not None else "n/a"
            lines.append(f"| {a} | {n} | {asr:.2%} | {ci_str} | {avg_q:.1f} | {avg_ed:.3f} |")
        lines.append("")
        lines.append("![ASR by attack](figures/asr_by_attack.png)\n")

        if stats and stats.get("pairwise_mcnemar"):
            lines.append("### Statistical significance between attacks (McNemar's test)\n")
            lines.append("| Comparison | statistic | p-value | significant (a=0.05) |")
            lines.append("|---|---|---|---|")
            for comp, result in stats["pairwise_mcnemar"].items():
                sig = result.get("significant_at_0.05", "n/a")
                lines.append(f"| {comp} | {result.get('statistic')} | {result.get('p_value')} | {sig} |")
            lines.append("")

    # --- Section 2: cross-model transfer --------------------------------
    matrix = _load_json(config.SURROGATE_TRANSFER_MATRIX_PATH)
    if matrix:
        lines.append("## 2. Cross-Model Transfer (surrogate -> surrogate)\n")
        lines.append(
            "Fraction of attacks that succeed against a *different* guardrail "
            "model than the one they were optimized against.\n"
        )
        lines.append("![Cross-model transfer](figures/cross_model_transfer.png)\n")

    # --- Section 3: black-box API transfer ------------------------------
    api_transfer = _load_json(config.TRANSFER_RESULTS_PATH)
    if api_transfer:
        lines.append("## 3. Black-Box API Transfer\n")
        lines.append("| API target | n tested | n transferred | transfer rate |")
        lines.append("|---|---|---|---|")
        for name, r in api_transfer.items():
            lines.append(f"| {name} | {r['n_tested']} | {r['n_transferred']} | {r['transfer_rate']} |")
        lines.append("")

    # --- Section 4: ablations --------------------------------------------
    ablations = _load_json(config.ABLATION_RESULTS_PATH)
    if ablations:
        lines.append("## 4. Ablations\n")
        lines.append("![Query budget ablation](figures/ablation_query_budget.png)\n")
        lines.append("![Similarity threshold ablation](figures/ablation_similarity_threshold.png)\n")

    # --- Section 5: perturbation cost -------------------------------------
    if records:
        lines.append("## 5. Perturbation Cost\n")
        lines.append("![Edit distance vs success](figures/edit_distance_vs_success.png)\n")

    # --- Section 6: defense --------------------------------------------
    defense = _load_json(config.DEFENSE_RESULTS_PATH)
    if defense:
        lines.append("## 6. Defense: Adversarial Fine-Tuning\n")
        lines.append("![Defense before/after](figures/defense_before_after.png)\n")
        before_overall = defense["before_defense"].get("overall", {})
        after_overall = defense["after_defense"].get("overall", {})
        if before_overall.get("n") and after_overall.get("n"):
            lines.append(
                f"Overall ASR before defense: **{before_overall['attack_success_rate']:.2%}** "
                f"(n={before_overall['n']}). Overall ASR after adversarial fine-tuning: "
                f"**{after_overall['attack_success_rate']:.2%}** (n={after_overall['n']}).\n"
            )
        clean = defense.get("clean_data_evaluation")
        if clean:
            lines.append(
                "### Clean-data cost of the defense\n\n"
                "A defense that drops ASR by flagging everything is not a real "
                "improvement -- this table reports whether the fine-tuned "
                "classifier still tells toxic and benign text apart.\n"
            )
            lines.append("![Clean-data trade-off](figures/defense_clean_data_tradeoff.png)\n")
            b, a = clean["before_defense"], clean["after_defense"]
            lines.append("| Metric | Before | After |")
            lines.append("|---|---|---|")
            lines.append(f"| Accuracy | {b['accuracy']:.2%} | {a['accuracy']:.2%} |")
            lines.append(f"| False Positive Rate | {b['false_positive_rate']:.2%} | {a['false_positive_rate']:.2%} |")
            lines.append(f"| False Negative Rate | {b['false_negative_rate']:.2%} | {a['false_negative_rate']:.2%} |")
            lines.append("")

    lines.append("## Ethics & Responsible Use\n")
    lines.append(
        "All attacks were evaluated against classifiers only; no harmful "
        "content was published anywhere. Seed text is drawn from existing "
        "public toxicity-classification datasets. Findings involving any "
        "third-party production API should be responsibly disclosed to the "
        "relevant provider before publication.\n"
    )

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"Report written to {config.REPORT_PATH}")


if __name__ == "__main__":
    main()
