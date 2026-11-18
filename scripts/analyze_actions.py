"""Audit canonical action labels in DriveLM behavior answers."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from drivelm.action_labels import normalize_behavior_action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", default="data/processed/train.jsonl", nargs="?")
    parser.add_argument("--show-unmapped", type=int, default=20)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    unmapped: list[str] = []
    total = 0
    with Path(args.jsonl).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("task") != "behavior":
                continue
            total += 1
            action = normalize_behavior_action(row["answer"])
            if action is None:
                if len(unmapped) < args.show_unmapped:
                    unmapped.append(row["answer"])
            else:
                counts[action.value] += 1

    print(f"Behavior samples: {total:,}")
    print(f"Mapped: {sum(counts.values()):,} ({sum(counts.values()) / max(1, total):.1%})")
    print(f"Unmapped: {total - sum(counts.values()):,}")
    for action, count in sorted(counts.items()):
        print(f"{action:22} {count:,}")
    if unmapped:
        print("\nUnmapped examples:")
        for answer in unmapped:
            print(f"- {answer}")


if __name__ == "__main__":
    main()

