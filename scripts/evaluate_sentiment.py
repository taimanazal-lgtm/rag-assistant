"""Compare VADER sentiment with the hand-labeled sample."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

LABELS = ["positive", "neutral", "negative"]
OUT = Path("results/sentiment_eval.json")


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def main():
    gold = pd.read_csv("labels/sentiment_labels.csv")
    gold["label"] = gold["label"].astype(str).str.strip().str.lower()
    bad = gold[~gold["label"].isin(LABELS)]
    if len(bad):
        sys.exit(f"{len(bad)} rows have a missing or invalid label (use positive/neutral/negative).")

    pred = pd.read_parquet("data/processed/doc_nlp.parquet").set_index("id")["sentiment"]
    y_true, y_pred = gold["label"], gold["id"].map(pred)

    n = len(gold)
    acc = accuracy_score(y_true, y_pred)
    low, high = wilson(int((y_true == y_pred).sum()), n)
    baseline = float(y_true.value_counts(normalize=True).max())
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    print(f"n = {n}")
    print(f"Accuracy: {acc:.1%} (95% interval {low:.1%} to {high:.1%})")
    print(f"Majority-class baseline: {baseline:.1%} | macro F1: {macro_f1:.2f}\n")
    print("Confusion matrix (rows = my label, columns = VADER):")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))
    print()
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    OUT.write_text(json.dumps({
        "n": n,
        "accuracy": round(acc, 4),
        "accuracy_95_interval": [round(low, 4), round(high, 4)],
        "majority_baseline": round(baseline, 4),
        "macro_f1": round(float(macro_f1), 4),
        "label_counts": {k: int(v) for k, v in y_true.value_counts().items()},
        "confusion_matrix": {"labels": LABELS, "rows_gold_cols_vader": cm.tolist()},
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()