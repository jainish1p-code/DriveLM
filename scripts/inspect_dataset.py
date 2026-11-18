"""Inspect one processed DriveLM sample and verify all camera files."""

from __future__ import annotations

import argparse

from drivelm.qwen_dataset import CAMERA_ORDER, DriveLMQADataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="?", default="data/processed/train.jsonl")
    parser.add_argument("--image-root", default="data/nuscenes")
    args = parser.parse_args()

    dataset = DriveLMQADataset(args.jsonl, args.image_root)
    sample = dataset[0]
    print(f"Dataset rows: {len(dataset):,}")
    print(f"Scene: {sample.scene_token}")
    print(f"Frame: {sample.frame_token}")
    print(f"Task: {sample.task}")
    print(f"Prompt: {sample.prompt}")
    print(f"Answer: {sample.answer}")
    print("Images:")
    for camera, path in zip(CAMERA_ORDER, sample.image_paths):
        print(f"  {camera}: {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

