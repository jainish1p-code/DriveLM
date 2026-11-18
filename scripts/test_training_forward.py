"""Run one Qwen2-VL forward pass and report the training loss."""

from __future__ import annotations

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

from drivelm.collator import DriveLMQwenCollator
from drivelm.qwen_dataset import DriveLMQADataset


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


def main() -> None:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    dataset = DriveLMQADataset("data/processed/train.jsonl", "data/nuscenes")
    batch = DriveLMQwenCollator(processor)([dataset[0]])
    batch = {key: value.to("cuda") if hasattr(value, "to") else value for key, value in batch.items()}

    with torch.no_grad():
        output = model(**batch)

    print(f"Loss: {output.loss.item():.4f}")
    print(f"GPU allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()

