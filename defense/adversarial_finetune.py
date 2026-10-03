"""
Defense experiment: adversarially fine-tune the local surrogate classifier on
a mix of clean + successfully-attacked examples (labeled as their true
class), then re-run the attack evaluation to measure whether ASR drops.

This is the mitigation half of the paper -- every attack result should be
paired with a "does a cheap defense help?" result.

Usage:
    python defense/adversarial_finetune.py --results results/attack_results.jsonl
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.load_data import load_dataset
from targets.local_classifier import LocalSurrogateClassifier
from evaluate import load_results, run_attacks, summarize, save_results


def build_adversarial_training_set(records, benign_pool=None, extra_neg_ratio: float = 1.0,
                                   seed: int = None):
    """
    Builds a fine-tuning set from successful adversarial examples (correct
    label = the ORIGINAL label, since the perturbation was meaning-preserving
    and the model was fooled into flipping its prediction), plus the
    original clean examples, plus BENIGN examples.

    The benign examples matter: attack records only contain toxic texts, so
    training on records alone teaches the model "everything is toxic" --
    which kills ASR trivially while making the classifier useless (exactly
    the failure mode the clean-data FPR check is there to catch).
    `extra_neg_ratio` = benign examples per toxic example.
    """
    seed = seed if seed is not None else config.RANDOM_SEED
    successes = [r for r in records if r["success"]]
    # dedupe: each clean text appears once per attack in the records file
    clean = {r["orig_text"]: r["orig_label"] for r in records}
    adv = {r["best_text"]: r["orig_label"] for r in successes}
    examples = list(clean.items()) + [(t, l) for t, l in adv.items() if t not in clean]

    n_pos = sum(1 for _, l in examples if l == 1)
    if benign_pool:
        seen = {t for t, _ in examples}
        benign = [(t, 0) for t, l in benign_pool if l == 0 and t not in seen]
        import random
        random.Random(seed).shuffle(benign)
        examples += benign[: int(n_pos * extra_neg_ratio)]
    return examples


def finetune_classifier(classifier: LocalSurrogateClassifier, training_data):
    """
    Fine-tunes the underlying HF model on the adversarial training set.
    Requires transformers + torch and enough local compute for a few epochs
    over a small dataset (CPU-feasible given dataset sizes used here).
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    import torch
    from torch.utils.data import Dataset

    from transformers import AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(classifier.model_name)
    # Force a binary, single-label (softmax + cross-entropy) head with
    # explicit label names. unitary/toxic-bert ships a 6-label MULTI-label
    # head (problem_type="multi_label_classification"); keeping that
    # problem_type with integer 0/1 labels makes the Trainer's BCE loss
    # crash on a shape mismatch, and unnamed LABEL_0/LABEL_1 outputs would
    # break the toxic-label scoring in LocalSurrogateClassifier.
    base_cfg = AutoConfig.from_pretrained(classifier.model_name)
    reuse_head = base_cfg.num_labels == 2 and base_cfg.problem_type != "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        classifier.model_name,
        num_labels=2,
        problem_type="single_label_classification",
        id2label={0: "non-toxic", 1: "toxic"},
        label2id={"non-toxic": 0, "toxic": 1},
        ignore_mismatched_sizes=not reuse_head,
    )

    class SimpleDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text, label = self.data[idx]
            enc = tokenizer(text, truncation=True, padding="max_length", max_length=64)
            enc = {k: torch.tensor(v) for k, v in enc.items()}
            enc["labels"] = torch.tensor(label)
            return enc

    dataset = SimpleDataset(training_data)

    args = TrainingArguments(
        output_dir="defense/finetune_out",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    model.save_pretrained("defense/finetuned_model")
    tokenizer.save_pretrained("defense/finetuned_model")
    return "defense/finetuned_model"


def evaluate_clean_data_performance(classifier, eval_data) -> dict:
    """
    Evaluates the classifier's standard classification performance on a
    labeled, UNPERTURBED evaluation set -- both toxic (label=1) and benign
    (label=0) examples. This is what tells you whether a defense "worked" by
    genuinely improving robustness, or just by flagging everything (which
    would trivially kill ASR while making the classifier useless).

    Returns accuracy, false positive rate (benign examples wrongly flagged),
    and false negative rate (toxic examples wrongly missed).
    """
    benign = [(t, l) for t, l in eval_data if l == 0]
    toxic = [(t, l) for t, l in eval_data if l == 1]

    correct = 0
    for text, label in eval_data:
        pred = int(classifier.is_flagged(text))
        correct += int(pred == label)
    accuracy = correct / len(eval_data) if eval_data else None

    fp = sum(1 for t, _ in benign if classifier.is_flagged(t))
    fpr = fp / len(benign) if benign else None

    fn = sum(1 for t, _ in toxic if not classifier.is_flagged(t))
    fnr = fn / len(toxic) if toxic else None

    return {
        "n_total": len(eval_data),
        "n_benign": len(benign),
        "n_toxic": len(toxic),
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "false_positive_rate": round(fpr, 4) if fpr is not None else None,
        "false_negative_rate": round(fnr, 4) if fnr is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=config.ATTACK_RESULTS_PATH)
    parser.add_argument("--n_eval_samples", type=int, default=100)
    parser.add_argument(
        "--n_clean_eval_samples", type=int, default=150,
        help="Size of the fresh, unperturbed, labeled set used to measure "
             "clean-data accuracy / false-positive rate before vs after "
             "fine-tuning -- must be disjoint from the training data to be "
             "meaningful, hence drawn fresh with a different seed.",
    )
    args = parser.parse_args()

    records = load_results(args.results)
    benign_pool = load_dataset(n_samples=4 * len(records) + 50, seed=config.RANDOM_SEED + 500)
    training_data = build_adversarial_training_set(records, benign_pool=benign_pool)
    train_texts = {t for t, _ in training_data}
    n_tox = sum(1 for _, l in training_data if l == 1)
    print(f"Built adversarial training set with {len(training_data)} examples "
          f"({n_tox} toxic incl. {sum(1 for r in records if r['success'])} adversarial "
          f"successes, {len(training_data) - n_tox} benign).")

    base_classifier = LocalSurrogateClassifier()

    # Fresh, disjoint, labeled eval set (both classes) for clean-data metrics.
    # Different seed from the main experiment so it doesn't overlap with the
    # training data or the attack-eval data.
    clean_eval_data = [
        p for p in load_dataset(n_samples=args.n_clean_eval_samples, seed=config.RANDOM_SEED + 999)
        if p[0] not in train_texts
    ]

    print("\n=== Clean-data performance BEFORE defense ===")
    clean_before = evaluate_clean_data_performance(base_classifier, clean_eval_data)
    print(json.dumps(clean_before, indent=2))

    finetuned_path = finetune_classifier(base_classifier, training_data)
    print(f"Fine-tuned model saved to {finetuned_path}")

    finetuned_classifier = LocalSurrogateClassifier(model_name=finetuned_path)

    print("\n=== Clean-data performance AFTER defense ===")
    clean_after = evaluate_clean_data_performance(finetuned_classifier, clean_eval_data)
    print(json.dumps(clean_after, indent=2))
    if clean_before["false_positive_rate"] is not None and clean_after["false_positive_rate"] is not None:
        fpr_delta = clean_after["false_positive_rate"] - clean_before["false_positive_rate"]
        print(f"False-positive rate change: {fpr_delta:+.2%} "
              f"({'WARNING: defense increased FPR meaningfully' if fpr_delta > 0.05 else 'acceptable'})")

    print("\nRe-running attack evaluation against the fine-tuned model...")
    # Fresh examples the fine-tuned model never trained on (the original
    # seed's sample IS the training data, so re-attacking it would leak).
    eval_data = [
        p for p in load_dataset(n_samples=args.n_eval_samples, seed=config.RANDOM_SEED + 2024)
        if p[0] not in train_texts
    ]
    attack_names = sorted({r["attack"] for r in records})
    new_records = run_attacks(eval_data, finetuned_classifier, attack_names=attack_names)
    new_summary = summarize(new_records)

    # Apples-to-apples baseline: attack the ORIGINAL model on the SAME fresh
    # examples (the records file covers different examples).
    print("Re-running the same attacks against the original model on the same examples...")
    before_summary = summarize(run_attacks(eval_data, base_classifier, attack_names=attack_names))

    comparison = {
        "before_defense": before_summary,
        "after_defense": new_summary,
        "clean_data_evaluation": {
            "before_defense": clean_before,
            "after_defense": clean_after,
        },
    }
    print(json.dumps(comparison, indent=2))

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.DEFENSE_RESULTS_PATH, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved defense before/after comparison to {config.DEFENSE_RESULTS_PATH}")


if __name__ == "__main__":
    main()
