"""Read the perforated passport field from an image with a compatible detector and CRNN."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.localization.dinov2 import DINOv2Locator
from src.localization.dot_span import dot_span_recrop
from src.recognition.perforation import PerforationRecognizer


def predict(image, detector, reader):
    box, score = detector.locate(image)
    result = dict(bbox=box, localization_score=score, readable=False)
    if not box:
        result["review_reason"] = "field_not_found"
        return result

    checked = dot_span_recrop(image, box, reader)
    prediction = checked.prediction
    result.update(
        bbox=checked.box, detector_box=box, baseline_box=checked.baseline_box,
        baseline_prediction=checked.baseline_prediction.raw_text, refinement=checked.status,
        geometry_review=checked.status in ("dot_rows_unverified", "disputed_dot_span"),
        raw_text=prediction.raw_text, confidence=prediction.confidence,
        character_scores=prediction.character_scores, readable=prediction.readable,
        review_reason=prediction.review_reason,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.image, args.detector, args.reader):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    torch.set_num_threads(2)
    image = cv2.imdecode(np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        parser.error("Image could not be decoded")
    result = predict(image, DINOv2Locator(args.detector), PerforationRecognizer(args.reader))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
