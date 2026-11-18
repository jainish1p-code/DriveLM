from drivelm.dataset import flatten_annotations, split_by_scene


def sample_annotations():
    return {
        "scene-a": {
            "scene_description": "The ego vehicle proceeds straight.",
            "key_frames": {
                "frame-a": {
                    "key_object_infos": {"<c1,CAM_FRONT,10,20>": {"Category": "Vehicle"}},
                    "image_paths": {"CAM_FRONT": "samples/front.jpg"},
                    "QA": {
                        "perception": [{"Q": "What is ahead?", "A": "A vehicle."}],
                        "prediction": [],
                        "planning": [],
                        "behavior": [],
                    },
                }
            },
        },
        "scene-b": {
            "scene_description": "The ego vehicle turns.",
            "key_frames": {
                "frame-b": {
                    "key_object_infos": {},
                    "image_paths": {},
                    "QA": {task: [] for task in ("perception", "prediction", "planning", "behavior")},
                }
            },
        },
    }


def test_flatten_annotations_preserves_scene_and_task():
    samples = flatten_annotations(sample_annotations())

    assert len(samples) == 1
    assert samples[0].scene_token == "scene-a"
    assert samples[0].task == "perception"
    assert samples[0].image_paths["CAM_FRONT"] == "samples/front.jpg"


def test_split_by_scene_is_deterministic_and_disjoint():
    samples = flatten_annotations(sample_annotations())
    splits = split_by_scene(samples, train_ratio=0.5, validation_ratio=0.25, seed=7)

    scene_sets = [{sample.scene_token for sample in items} for items in splits.values()]
    assert sum(len(scene_set) for scene_set in scene_sets) == len(set().union(*scene_sets))

