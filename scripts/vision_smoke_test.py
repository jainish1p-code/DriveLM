"""Run a minimal Qwen2-VL image-understanding smoke test.

This script validates that the model, quantization configuration, processor,
GPU, and remote image loading work before we connect DriveLM data.
"""

from __future__ import annotations

import torch
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)


MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEMO_IMAGE = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"


def load_model():
    """Load Qwen2-VL in 4-bit mode for the A100 training environment."""

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
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        size={
            "shortest_edge": 256 * 28 * 28,
            "longest_edge": 768 * 28 * 28,
        },
    )
    return model, processor


def run_smoke_test() -> str:
    """Ask the model to describe the official Qwen demonstration image."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run this on the Jetstream2 GPU instance")

    model, processor = load_model()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": DEMO_IMAGE},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=80)
    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output[0]


if __name__ == "__main__":
    print("Loading Qwen2-VL smoke test...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Response: {run_smoke_test()}")

