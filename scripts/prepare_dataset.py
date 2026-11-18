"""Prepare flattened, scene-disjoint DriveLM-nuScenes JSONL files."""

from __future__ import annotations

import argparse
from pathlib import Path

from drivelm.dataset import flatten_annotations, load_json, split_by_scene, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = flatten_annotations(load_json(args.annotation_file))
    splits = split_by_scene(samples, seed=args.seed)
    print(f"Loaded {len(samples):,} QA samples from {args.annotation_file}")
    for split, split_samples in splits.items():
        output = args.output_dir / f"{split}.jsonl"
        count = write_jsonl(split_samples, output)
        print(f"{split:>10}: {count:,} samples -> {output}")


if __name__ == "__main__":
    main()

