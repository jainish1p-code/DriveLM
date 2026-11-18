"""DriveLM-nuScenes annotation parsing and scene-level splitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable


TASKS = ("perception", "prediction", "planning", "behavior")


@dataclass(frozen=True)
class QASample:
    """One flattened question-answer example from a DriveLM key frame."""

    scene_token: str
    frame_token: str
    scene_description: str
    task: str
    question: str
    answer: str
    image_paths: dict[str, str]
    key_object_infos: dict[str, Any]
    context: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _qa_value(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def flatten_annotations(annotation_data: dict[str, Any]) -> list[QASample]:
    """Flatten the nested official JSON structure into training samples."""

    samples: list[QASample] = []
    for scene_token, scene in annotation_data.items():
        if not isinstance(scene, dict):
            continue
        scene_description = str(scene.get("scene_description", ""))
        key_frames = scene.get("key_frames", {})
        if not isinstance(key_frames, dict):
            continue
        for frame_token, frame in key_frames.items():
            if not isinstance(frame, dict):
                continue
            qa_by_task = frame.get("QA", {})
            if not isinstance(qa_by_task, dict):
                continue
            image_paths = frame.get("image_paths", {})
            key_objects = frame.get("key_object_infos", {})
            for task in TASKS:
                qa_items = qa_by_task.get(task, [])
                if not isinstance(qa_items, list):
                    continue
                for item in qa_items:
                    if not isinstance(item, dict):
                        continue
                    question = _qa_value(item, "Q", "question")
                    answer = _qa_value(item, "A", "answer")
                    if question is None or answer is None:
                        continue
                    samples.append(
                        QASample(
                            scene_token=str(scene_token),
                            frame_token=str(frame_token),
                            scene_description=scene_description,
                            task=task,
                            question=str(question),
                            answer=str(answer),
                            image_paths=dict(image_paths) if isinstance(image_paths, dict) else {},
                            key_object_infos=dict(key_objects) if isinstance(key_objects, dict) else {},
                            context=item.get("C"),
                        )
                    )
    return samples


def split_by_scene(
    samples: Iterable[QASample],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[QASample]]:
    """Split samples without allowing frames from one scene to cross splits."""

    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio and validation_ratio must be positive and sum to less than 1")

    grouped: dict[str, list[QASample]] = {}
    for sample in samples:
        grouped.setdefault(sample.scene_token, []).append(sample)

    scenes = list(grouped)
    random.Random(seed).shuffle(scenes)
    train_end = round(len(scenes) * train_ratio)
    validation_end = round(len(scenes) * (train_ratio + validation_ratio))
    assignments = {
        "train": scenes[:train_end],
        "validation": scenes[train_end:validation_end],
        "test": scenes[validation_end:],
    }
    return {
        split: [sample for scene in scene_tokens for sample in grouped[scene]]
        for split, scene_tokens in assignments.items()
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at the root of {path}")
    return data


def write_jsonl(samples: Iterable[QASample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count

