"""Run a controlled QLoRA training smoke test on DriveLM-nuScenes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from drivelm.collator import DriveLMQwenCollator
from drivelm.qwen_dataset import DriveLMQADataset


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", default="data/processed/train.jsonl")
    parser.add_argument("--image-root", default="data/nuscenes")
    parser.add_argument("--output-dir", default="checkpoints/qwen2vl-drivelm-smoke")
    parser.add_argument("--init-adapter", default=None)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument(
        "--task",
        choices=("perception", "prediction", "planning", "behavior"),
        default=None,
    )
    parser.add_argument("--balance-actions", action="store_true")
    parser.add_argument("--balance-tasks", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def move_to_cuda(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value.to("cuda", non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.manual_seed(args.seed)

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
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    if args.init_adapter:
        model = PeftModel.from_pretrained(
            model,
            args.init_adapter,
            is_trainable=True,
        )
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=TARGET_MODULES,
            ),
        )
    model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    dataset = DriveLMQADataset(args.train_jsonl, args.image_root)
    candidate_indices = [
        index
        for index, row in enumerate(dataset._rows)
        if args.task is None or row.get("task") == args.task
    ]
    limit = len(candidate_indices) if args.max_samples <= 0 else min(args.max_samples, len(candidate_indices))
    dataset = Subset(dataset, candidate_indices[:limit])
    collator = DriveLMQwenCollator(processor)
    sampler = None
    if args.balance_actions and args.balance_tasks:
        raise ValueError("Choose only one of --balance-actions or --balance-tasks")
    if args.balance_actions:
        if args.task != "behavior":
            raise ValueError("--balance-actions requires --task behavior")
        labels = [dataset[index].answer for index in range(len(dataset))]
        frequencies = {label: labels.count(label) for label in set(labels)}
        weights = [1.0 / frequencies[label] for label in labels]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
    elif args.balance_tasks:
        if args.task is not None:
            raise ValueError("--balance-tasks is for multi-task training; omit --task")
        labels = [dataset[index].task for index in range(len(dataset))]
        frequencies = {label: labels.count(label) for label in set(labels)}
        weights = [1.0 / frequencies[label] for label in labels]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collator,
        pin_memory=True,
    )

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    completed_steps = 0
    micro_steps = 0
    running_loss = 0.0

    while completed_steps < args.max_steps:
        for batch in loader:
            batch = move_to_cuda(batch)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
                scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            running_loss += loss.detach().item()
            micro_steps += 1

            if micro_steps % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed_steps += 1
                print(
                    f"step={completed_steps}/{args.max_steps} "
                    f"loss={running_loss / args.gradient_accumulation_steps:.4f} "
                    f"gpu_peak={torch.cuda.max_memory_allocated() / 1024**3:.2f}GB",
                    flush=True,
                )
                running_loss = 0.0
                torch.cuda.reset_peak_memory_stats()
                if completed_steps >= args.max_steps:
                    break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()

