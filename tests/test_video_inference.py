import numpy as np

from src.video_inference import draw_annotations_from_results


def test_draw_annotations_from_results_adds_boxes_and_labels():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    results = [
        {
            "box": (10, 10, 40, 40),
            "label": "Happiness",
            "confidence": 0.95,
        }
    ]

    annotated = draw_annotations_from_results(frame, results)

    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)
    assert annotated[10, 10].tolist() != frame[10, 10].tolist()
