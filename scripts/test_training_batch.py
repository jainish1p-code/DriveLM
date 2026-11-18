"""Build one six-camera Qwen2-VL training batch without loading model weights."""

from __future__ import annotations

import torch
from transformers import AutoProcessor

from drivelm.collator import DriveLMQwenCollator
from drivelm.qwen_dataset import DriveLMQADataset


def main() -> None:
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    dataset = DriveLMQADataset(
        "data/processed/train.jsonl",
        "data/nuscenes",
    )
    batch = DriveLMQwenCollator(processor)([dataset[0]])

    print(f"CUDA available: {torch.cuda.is_available()}")
    for name, value in batch.items():
        if hasattr(value, "shape"):
            print(f"{name}: {tuple(value.shape)}")
        else:
            print(f"{name}: {type(value).__name__}")


if __name__ == "__main__":
    main()

