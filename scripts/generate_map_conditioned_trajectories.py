"""Generate map-conditioned planner trajectories for model predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from nuscenes.map_expansion.map_api import NuScenesMap

from drivelm.action_labels import normalize_behavior_action
from drivelm.planner import plan_reference_trajectory


def load_table(root: Path, name: str) -> list[dict]:
    with (root / f"{name}.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def yaw_from_quaternion(rotation: list[float]) -> float:
    w, x, y, z = rotation
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def resolve_sample(frame_token: str, samples: dict, sample_data: dict) -> dict:
    if frame_token in samples:
        return samples[frame_token]
    return samples[sample_data[frame_token]["sample_token"]]


def local_lane_path(nusc_map, pose: dict, radius: float) -> tuple[str, list[dict[str, float]]]:
    ego_x, ego_y = pose["translation"][:2]
    ego_yaw = yaw_from_quaternion(pose["rotation"])
    nearby = nusc_map.get_records_in_radius(
        ego_x,
        ego_y,
        radius,
        ["lane", "lane_connector"],
    )
    candidates = nearby["lane"] + nearby["lane_connector"]
    if not candidates:
        raise RuntimeError(f"No lane found within {radius} m")
    lane_paths = nusc_map.discretize_lanes(candidates, resolution_meters=0.5)

    def angle_error(left: float, right: float) -> float:
        return abs(math.atan2(math.sin(left - right), math.cos(left - right)))

    ranked = []
    for candidate in candidates:
        path = lane_paths[candidate]
        index = min(
            range(len(path)),
            key=lambda item: math.hypot(path[item][0] - ego_x, path[item][1] - ego_y),
        )
        point = path[index]
        distance = math.hypot(point[0] - ego_x, point[1] - ego_y)
        heading_error = angle_error(point[2], ego_yaw)
        ranked.append((distance + 5.0 * heading_error, candidate, path, index))
    _, lane_token, global_path, closest_index = min(ranked, key=lambda item: item[0])
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
    if len(local_path) == 1:
        last = local_path[0]
        yaw = last.get("yaw", 0.0)
        local_path.append(
            {
                "x": last["x"] + 40.0 * math.cos(yaw),
                "y": last["y"] + 40.0 * math.sin(yaw),
                "yaw": yaw,
            }
        )
    return lane_token, local_path


def speed_from_previous(sample: dict, samples: dict, front_data: dict, poses: dict) -> float:
    if not sample.get("prev"):
        return 10.0
    previous = samples[sample["prev"]]
    current_pose = poses[front_data[sample["token"]]["ego_pose_token"]]
    previous_pose = poses[front_data[previous["token"]]["ego_pose_token"]]
    dt = (sample["timestamp"] - previous["timestamp"]) / 1_000_000
    if dt <= 0:
        return 10.0
    dx = current_pose["translation"][0] - previous_pose["translation"][0]
    dy = current_pose["translation"][1] - previous_pose["translation"][1]
    return math.hypot(dx, dy) / dt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/map_conditioned_trajectories.jsonl"))
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
    map_cache = {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted = 0
    with args.predictions.open(encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            row = json.loads(line)
            action = normalize_behavior_action(row["prediction"])
            if action is None:
                continue
            sample = resolve_sample(row["frame_token"], samples, sample_data)
            scene = scene_rows[sample["scene_token"]]
            location = log_rows[scene["log_token"]]["location"]
            if location not in map_cache:
                map_cache[location] = NuScenesMap(dataroot=str(args.map_root), map_name=location)
            pose = poses[front_data[sample["token"]]["ego_pose_token"]]
            lane_token, reference_path = local_lane_path(map_cache[location], pose, args.radius)
            speed = speed_from_previous(sample, samples, front_data, poses)
            trajectory = plan_reference_trajectory(action, reference_path, current_speed=speed)
            destination.write(
                json.dumps(
                    {
                        **row,
                        "action": action.value,
                        "location": location,
                        "lane_token": lane_token,
                        "initial_speed_from_ego_pose_mps": speed,
                        "reference_path": reference_path,
                        "trajectory": [point.__dict__ for point in trajectory],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            converted += 1

    print(f"Converted map-conditioned trajectories: {converted}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
