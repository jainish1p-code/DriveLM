"""Batch collation for Qwen2-VL DriveLM conversations."""

from __future__ import annotations

from typing import Any

import torch
from qwen_vl_utils import process_vision_info

from .qwen_dataset import ConversationExample


class DriveLMQwenCollator:
    """Convert DriveLM examples into Qwen2-VL tensors.

    The initial training configuration uses batch size one because each sample
    contains six camera images. The implementation supports larger batches
    when GPU memory permits.
    """

    def __init__(self, processor: Any, max_length: int = 16384):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, examples: list[ConversationExample]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        images: list[Any] = []
        videos: list[Any] = []
        target_token_ids: list[list[int]] = []

        for example in examples:
            messages = example.messages()
            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            image_inputs, video_inputs = process_vision_info(messages)
            images.extend(image_inputs or [])
            videos.extend(video_inputs or [])
            target_token_ids.append(
                self.processor.tokenizer(
                    example.answer,
                    add_special_tokens=False,
                )["input_ids"]
            )

        batch = self.processor(
            text=texts,
            images=images or None,
            videos=videos or None,
            padding=True,
            # Do not truncate below the number of visual placeholder tokens.
            # Six camera views can already exceed 2,048 tokens.
            truncation=False,
            return_tensors="pt",
        )
        labels = torch.full_like(batch["input_ids"], -100)
        for row_index, target_ids in enumerate(target_token_ids):
            input_ids = batch["input_ids"][row_index].tolist()
            start = None
            for candidate in range(len(input_ids) - len(target_ids), -1, -1):
                if input_ids[candidate : candidate + len(target_ids)] == target_ids:
                    start = candidate
                    break
            if start is None:
                # Tokenization can differ at a chat-template boundary. The
                # assistant answer is the final non-padding segment, so use
                # its token length as a safe fallback.
                valid_length = int(batch["attention_mask"][row_index].sum().item())
                start = max(0, valid_length - len(target_ids))
            end = min(start + len(target_ids), labels.shape[1])
            labels[row_index, start:end] = batch["input_ids"][row_index, start:end]
        batch["labels"] = labels
        return batch

