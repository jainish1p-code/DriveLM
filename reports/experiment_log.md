# DriveLM Experiment Log

Updated: 2026-11-19

## Environment

- Platform: Jetstream2 `g3.xl`
- GPU: NVIDIA A100-SXM4-40GB
- Model: `Qwen/Qwen2-VL-7B-Instruct`
- Fine-tuning: 4-bit QLoRA with PEFT
- Input: six nuScenes camera views
- Training image size: 224 x 224 per view

## Data

DriveLM v1.1 was flattened into scene-disjoint JSONL files:

| Split | QA samples |
|---|---:|
| Train | 264,582 |
| Validation | 57,657 |
| Test | 55,717 |
| Total | 377,956 |

The image subset contains 24,432 images across six cameras and occupies approximately 3.4 GB.

## Checkpoint comparison

Metrics below are preliminary: general language metrics use 100 validation samples, and behavior metrics use 50 targeted validation samples.

| Checkpoint | General F1 | General ROUGE-L | Behavior accuracy | Behavior macro-F1 |
|---|---:|---:|---:|---:|
| `qwen2vl-multitask-1000` | 0.669 | 0.665 | 0.480 | 0.251 |
| `qwen2vl-behavior-balanced-1000` | — | — | **0.600** | **0.524** |
| `qwen2vl-multitask-from-behavior-1000` | **0.718** | **0.714** | 0.500 | 0.364 |

## Finding

Initializing multi-task training from the behavior adapter improved general perception/prediction/planning language scores, but behavior-action performance remained below the behavior-only adapter. This indicates multi-task interference or catastrophic forgetting in the canonical action interface.

The current best general-reasoning checkpoint is `qwen2vl-multitask-from-behavior-1000`. The current best behavior-action checkpoint is `qwen2vl-behavior-balanced-1000`.

## Action-label caveat

The 10-class action labels are heuristic proxy labels derived from natural-language behavior descriptions. Seven actions are observed in the released validation data; `YIELD`, `LANE_CHANGE_LEFT`, and `LANE_CHANGE_RIGHT` are not represented in the observed behavior answers. These results should therefore be reported as an initial canonical-action experiment, not as a complete ground-truth benchmark for all 10 classes.

## Next experiments

1. Connect canonical behavior outputs to the deterministic Frenet-style planner.
2. Evaluate generated trajectories against ego pose/trajectory ground truth where available.
3. Run larger stratified validation sets and report confidence intervals.
4. Investigate behavior-preserving multi-task sampling or separate task adapters.
5. Add final plots, qualitative examples, report text, and presentation slides.

The next planner refinement is reference-path conditioning using the nuScenes map expansion. The current planner remains straight-road and does not yet use HD-map geometry.

## End-to-end planner artifact

The best behavior checkpoint was connected to the deterministic planner. On 50 held-out behavior predictions:

- 50/50 canonical actions were converted into trajectories.
- Each trajectory spans 3.0 seconds with 0.2-second sampling.
- The generated trajectories are stored in `reports/predicted_trajectories.jsonl`.
- A presentation figure is stored in `reports/predicted_trajectories.png`.

The current planner demonstration validates the action-to-trajectory interface. ADE/FDE evaluation remains pending because the downloaded DriveLM image subset and QA annotations do not include the ego-pose trajectory ground truth required for those metrics.

After adding the nuScenes `sample`, `sample_data`, and `ego_pose` metadata, the first 50-sample planner evaluation produced:

| Metric | Value |
|---|---:|
| ADE | 7.872 m |
| FDE | 15.062 m |

These values are from a deliberately simple baseline: every predicted trajectory used a fixed 10 m/s initial speed and a map-independent planner. They should not be interpreted as the final planner performance. The next refinement is to derive the initial ego speed from nuScenes poses and pass it into the action-conditioned planner.

Using the initial ego speed estimated from consecutive nuScenes ego poses reduced the error:

| Planner input | ADE | FDE |
|---|---:|---:|
| Fixed 10 m/s initial speed | 7.872 m | 15.062 m |
| Pose-derived initial speed | **3.151 m** | **6.971 m** |

The pose-derived version reduced ADE by approximately 60.0% and FDE by approximately 53.7% on the same 50 samples. This confirms that physically grounded state information materially improves the deterministic planner, even before adding map geometry or learned trajectory refinement.

The first map-conditioned planner ablation produced:

| Planner variant | ADE | FDE |
|---|---:|---:|
| Map path, position-only lane matching | 4.207 m | 8.191 m |
| Map path, distance + heading lane matching | 4.157 m | 8.031 m |

The heading-aware map matcher improved the map result slightly, but it remains worse than the pose-speed straight-road baseline. This suggests that nearest-lane selection and single-lane continuation are not yet sufficient for reliable map-conditioned planning at intersections. The map-conditioned result should be presented as an ablation, not as the current best planner.
