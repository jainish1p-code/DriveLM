"""Extract a local nuScenes lane centerline for one DriveLM frame."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from nuscenes.map_expansion.map_api import NuScenesMap


def load_table(root: Path, name: str) -> list[dict]:
    with (root / f"{name}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def yaw_from_quaternion(rotation: list[float]) -> float:
    w, x, y, z = rotation
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/map_reference_path.json"))
    parser.add_argument("--radius", type=float, default=50.0)
    args = parser.parse_args()

    scene_rows = {row["token"]: row for row in load_table(args.metadata_dir, "scene")}
    log_rows = {row["token"]: row for row in load_table(args.metadata_dir, "log")}
    samples = {row["token"]: row for row in load_table(args.metadata_dir, "sample")}
    sample_data_rows = load_table(args.metadata_dir, "sample_data")
    sample_data = {row["token"]: row for row in sample_data_rows}
    front_data = {
        row["sample_token"]: row
        for row in sample_data_rows
        if row.get("channel") == "CAM_FRONT" or "/CAM_FRONT/" in row.get("filename", "")
    }
    poses = {row["token"]: row for row in load_table(args.metadata_dir, "ego_pose")}

    with args.predictions.open(encoding="utf-8") as handle:
        first = json.loads(next(handle))
    frame_token = first["frame_token"]
    sample = (
        samples[frame_token]
        if frame_token in samples
        else samples[sample_data[frame_token]["sample_token"]]
    )
    scene = scene_rows[sample["scene_token"]]
    location = log_rows[scene["log_token"]]["location"]
    pose = poses[front_data[sample["token"]]["ego_pose_token"]]
    ego_x, ego_y = pose["translation"][:2]
    ego_yaw = yaw_from_quaternion(pose["rotation"])

    nusc_map = NuScenesMap(dataroot=str(args.map_root), map_name=location)
    lane_token = nusc_map.get_closest_lane(ego_x, ego_y, radius=args.radius)
    if lane_token is None:
        raise RuntimeError(f"No lane found within {args.radius} m at {location} ({ego_x}, {ego_y})")
    global_path = nusc_map.discretize_lanes([lane_token], resolution_meters=0.5)[lane_token]
    closest_index = min(
        range(len(global_path)),
        key=lambda index: math.hypot(global_path[index][0] - ego_x, global_path[index][1] - ego_y),
    )
    local_path = []
    for x, y, yaw in global_path[closest_index:]:
        dx, dy = x - ego_x, y - ego_y
        local_path.append(
            {
                "x": math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy,
                "y": -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy,
                "yaw": yaw - ego_yaw,
            }
        )
        if local_path[-1]["x"] >= 40.0:
            break

    artifact = {
        "scene_token": sample["scene_token"],
        "frame_token": frame_token,
        "location": location,
        "lane_token": lane_token,
        "reference_path": local_path,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Location: {location}")
    print(f"Lane: {lane_token}")
    print(f"Reference points: {len(local_path)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
