"""Lazy multimodal dataset adapter for DriveLM-nuScenes JSONL samples."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .action_labels import normalize_behavior_action

CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

ACTION_LABELS = (
    "STOP, YIELD, MAINTAIN_SPEED, ACCELERATE, DECELERATE, "
    "LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT, SLIGHT_LEFT_NUDGE, "
    "SLIGHT_RIGHT_NUDGE, TURNING_MANEUVER"
)


@dataclass(frozen=True)
class ConversationExample:
    """One Qwen2-VL conversation with resolved image paths."""

    scene_token: str
    frame_token: str
    task: str
    image_paths: tuple[Path, ...]
    prompt: str
    answer: str
    source_answer: str

    def messages(self) -> list[dict[str, Any]]:
        content: list[dict[str, str]] = [
            {
                "type": "image",
                # qwen-vl-utils opens local paths directly. Using Path.as_uri()
                # would percent-encode '+' in some official nuScenes filenames.
                "image": str(path),
                "resized_height": 224,
                "resized_width": 224,
            }
            for path in self.image_paths
        ]
        content.append({"type": "text", "text": self.prompt})
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": self.answer}]},
        ]


class DriveLMQADataset:
    """Memory-light JSONL reader that validates all six camera views."""

    def __init__(self, jsonl_path: str | Path, image_root: str | Path):
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        with self.jsonl_path.open(encoding="utf-8") as handle:
            self._rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self._rows)

    def _resolve_image(self, relative_path: str) -> Path:
        candidate = self.image_root / relative_path
        if candidate.exists():
            return candidate.resolve()
        # Some official annotations include ../nuscenes in the relative path.
        candidate = (self.image_root.parent / relative_path).resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Image does not exist: {relative_path}")

    def __getitem__(self, index: int) -> ConversationExample:
        row = self._rows[index]
        image_paths = row.get("image_paths", {})
        missing = [camera for camera in CAMERA_ORDER if camera not in image_paths]
        if missing:
            raise ValueError(f"Sample {index} is missing cameras: {missing}")

        resolved_paths = tuple(
            self._resolve_image(image_paths[camera]) for camera in CAMERA_ORDER
        )
        scene_description = row.get("scene_description", "")
        source_answer = str(row["answer"])
        answer = source_answer
        if row["task"] == "behavior":
            action = normalize_behavior_action(source_answer)
            if action is not None:
                answer = action.value
        prompt = (
            f"Driving task: {row['task']}.\n"
            f"Scene context: {scene_description}\n"
            f"Question: {row['question']}"
        )
        if row["task"] == "behavior":
            prompt += (
                "\nRespond with exactly one canonical action label from: "
                f"{ACTION_LABELS}."
            )
        return ConversationExample(
            scene_token=row["scene_token"],
            frame_token=row["frame_token"],
            task=row["task"],
            image_paths=resolved_paths,
            prompt=prompt,
            answer=answer,
            source_answer=source_answer,
        )

