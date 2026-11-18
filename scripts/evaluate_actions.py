"""Evaluate canonical behavior actions from generated prediction JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import json

from drivelm.action_labels import normalize_behavior_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=str)
    args = parser.parse_args()

    pairs: list[tuple[str, str | None]] = []
    unmapped_reference = 0
    unmapped_prediction = 0
    with open(args.predictions, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("task") != "behavior":
                continue
            reference = normalize_behavior_action(row["reference"])
            prediction = normalize_behavior_action(row["prediction"])
            if reference is None:
                unmapped_reference += 1
            if prediction is None:
                unmapped_prediction += 1
            if reference is not None:
                pairs.append((reference.value, prediction.value if prediction is not None else None))

    if not pairs:
        raise RuntimeError("No behavior samples with mapped reference and prediction actions")

    labels = sorted(pair[0] for pair in pairs)
    labels = sorted(set(labels))
    matrix = Counter(pairs)
    correct = sum(matrix[label, label] for label in labels)
    total = len(pairs)
    mapped = total - unmapped_prediction
    print(f"Behavior samples: {len(pairs)}")
    print(f"Unmapped references: {unmapped_reference}")
    print(f"Unmapped predictions: {unmapped_prediction}")
    print(f"Prediction coverage: {mapped / total:.3f}")
    print(f"Overall accuracy: {correct / total:.3f}")
    print(f"Accuracy among mapped predictions: {correct / mapped:.3f}" if mapped else "Accuracy among mapped predictions: n/a")
    print("\nPer-action metrics:")

    f1_scores = []
    for label in labels:
        true_positive = matrix[label, label]
        predicted = sum(matrix[reference, label] for reference in labels)
        actual = sum(matrix[label, prediction] for prediction in labels + [None])
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)
        print(
            f"{label:22} support={actual:3} "
            f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}"
        )
    print(f"Macro-F1: {sum(f1_scores) / len(f1_scores):.3f}")

    print("\nConfusion matrix (reference -> prediction):")
    print("reference\\prediction," + ",".join(labels + ["UNMAPPED"]))
    for reference in labels:
        values = [str(matrix[reference, prediction]) for prediction in labels]
        values.append(str(matrix[reference, None]))
        print(reference + "," + ",".join(values))


if __name__ == "__main__":
    main()

