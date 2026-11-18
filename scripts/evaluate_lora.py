"""Generate one answer from a trained DriveLM LoRA adapter."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

from drivelm.actions import Action
from drivelm.qwen_dataset import DriveLMQADataset


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(reference: str, prediction: str) -> float:
    reference_tokens = Counter(_tokens(reference))
    prediction_tokens = Counter(_tokens(prediction))
    overlap = sum((reference_tokens & prediction_tokens).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(prediction_tokens.values()))
    recall = overlap / max(1, sum(reference_tokens.values()))
    return 2 * precision * recall / (precision + recall)


def rouge_l(reference: str, prediction: str) -> float:
    reference_tokens = _tokens(reference)
    prediction_tokens = _tokens(prediction)
    table = [[0] * (len(prediction_tokens) + 1) for _ in range(len(reference_tokens) + 1)]
    for i, reference_token in enumerate(reference_tokens, start=1):
        for j, prediction_token in enumerate(prediction_tokens, start=1):
            if reference_token == prediction_token:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    lcs = table[-1][-1]
    if lcs == 0:
        return 0.0
    precision = lcs / max(1, len(prediction_tokens))
    recall = lcs / max(1, len(reference_tokens))
    return 2 * precision * recall / (precision + recall)


def _action_constraint(processor, prompt_length: int):
    action_token_sequences = [
        processor.tokenizer(action.value, add_special_tokens=False)["input_ids"]
        for action in Action
    ]
    eos_token_id = processor.tokenizer.eos_token_id

    def allowed_tokens(_batch_id: int, input_ids):
        generated = input_ids[prompt_length:].tolist()
        allowed = set()
        for sequence in action_token_sequences:
            if sequence[: len(generated)] == generated:
                if len(generated) < len(sequence):
                    allowed.add(sequence[len(generated)])
                elif eos_token_id is not None:
                    allowed.add(eos_token_id)
        return sorted(allowed) or [eos_token_id]

    return allowed_tokens


def generate_prediction(model, processor, example, max_new_tokens: int) -> str:
    user_messages = [example.messages()[0]]
    prompt = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(user_messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    if example.task == "behavior":
        generation_kwargs["prefix_allowed_tokens_fn"] = _action_constraint(
            processor, inputs.input_ids.shape[-1]
        )
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--jsonl", default="data/processed/validation.jsonl")
    parser.add_argument("--image-root", default="data/nuscenes")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--task",
        choices=("perception", "prediction", "planning", "behavior"),
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="reports/validation_predictions.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    dataset = DriveLMQADataset(args.jsonl, args.image_root)
    candidate_indices = [
        index
        for index, row in enumerate(dataset._rows)
        if args.task is None or row.get("task") == args.task
    ]
    if not candidate_indices:
        raise ValueError(f"No samples found for task={args.task!r}")
    if args.num_samples == 1:
        indices = [args.index]
        if args.task is not None and dataset._rows[args.index].get("task") != args.task:
            indices = [candidate_indices[0]]
    else:
        indices = random.Random(args.seed).sample(
            candidate_indices, min(args.num_samples, len(candidate_indices))
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    with output_path.open("w", encoding="utf-8") as output_file:
        for index in indices:
            example = dataset[index]
            prediction = generate_prediction(model, processor, example, args.max_new_tokens)
            result = {
                "index": index,
                "scene_token": example.scene_token,
                "frame_token": example.frame_token,
                "task": example.task,
                "question": example.prompt,
                "reference": example.source_answer,
                "target_action": example.answer if example.task == "behavior" else None,
                "prediction": prediction,
                "token_f1": token_f1(example.answer, prediction),
                "rouge_l": rouge_l(example.answer, prediction),
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            scores.append(result)
            print(
                f"[{len(scores)}/{len(indices)}] {example.task} "
                f"F1={result['token_f1']:.3f} ROUGE-L={result['rouge_l']:.3f}"
            )

    print(f"Saved predictions to {output_path}")
    print(f"Mean token F1: {sum(item['token_f1'] for item in scores) / len(scores):.3f}")
    print(f"Mean ROUGE-L: {sum(item['rouge_l'] for item in scores) / len(scores):.3f}")


if __name__ == "__main__":
    main()

