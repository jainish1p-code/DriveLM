import json

from drivelm.qwen_dataset import CAMERA_ORDER, DriveLMQADataset


def test_dataset_builds_six_camera_conversation(tmp_path):
    image_root = tmp_path / "nuscenes"
    image_root.mkdir()
    image_paths = {}
    for camera in CAMERA_ORDER:
        path = image_root / f"{camera}.jpg"
        path.write_bytes(b"image")
        image_paths[camera] = path.name

    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "scene_token": "scene",
                "frame_token": "frame",
                "scene_description": "Driving straight.",
                "task": "perception",
                "question": "What is ahead?",
                "answer": "A vehicle.",
                "image_paths": image_paths,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sample = DriveLMQADataset(jsonl, image_root)[0]
    assert len(sample.image_paths) == 6
    assert sample.messages()[0]["content"][-1]["text"].endswith("What is ahead?")
    assert sample.messages()[1]["content"][0]["text"] == "A vehicle."


def test_behavior_answer_uses_canonical_training_target(tmp_path):
    image_root = tmp_path / "nuscenes"
    image_root.mkdir()
    image_paths = {}
    for camera in CAMERA_ORDER:
        path = image_root / f"{camera}.jpg"
        path.write_bytes(b"image")
        image_paths[camera] = path.name

    jsonl = tmp_path / "behavior.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "scene_token": "scene",
                "frame_token": "frame",
                "scene_description": "Stopped at a light.",
                "task": "behavior",
                "question": "Predict the behavior.",
                "answer": "The ego vehicle is not moving.",
                "image_paths": image_paths,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sample = DriveLMQADataset(jsonl, image_root)[0]
    assert sample.answer == "STOP"
    assert sample.source_answer == "The ego vehicle is not moving."

