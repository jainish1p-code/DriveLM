from scripts.vision_smoke_test import DEMO_IMAGE, MODEL_ID


def test_smoke_test_configuration():
    assert MODEL_ID == "Qwen/Qwen2-VL-7B-Instruct"
    assert DEMO_IMAGE.startswith("https://")
    assert DEMO_IMAGE.endswith("demo.jpeg")

