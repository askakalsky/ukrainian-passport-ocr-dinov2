"""CPU recognition of an already localized perforated strip."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.preprocess import preprocess_for_checkpoint
from src.recognize import CONFIDENCE_THRESHOLD, load_recognizer, recognize_positions


@dataclass(frozen=True)
class PerforationResult:
    raw_text: str | None
    character_scores: tuple[float, ...]
    confidence: float
    review_reason: str | None

    @property
    def readable(self) -> bool:
        return self.review_reason is None


class PerforationRecognizer:
    def __init__(self, checkpoint: str | Path):
        self.model, self.metadata = load_recognizer(checkpoint)

    def recognize_crop(self, crop: np.ndarray) -> PerforationResult:
        """Accept a uint8 grayscale or BGR crop in the original polarity."""
        if not isinstance(crop, np.ndarray) or crop.dtype != np.uint8:
            raise ValueError("Expected a uint8 image")
        if crop.ndim != 2 and not (crop.ndim == 3 and crop.shape[2] == 3):
            raise ValueError("Expected a grayscale or BGR crop")
        if crop.size == 0:
            return PerforationResult(None, (), 0.0, "empty_crop")
        binary = preprocess_for_checkpoint(crop,self.metadata)
        if binary.min() == binary.max():
            return PerforationResult(None, (), 0.0, "no_foreground")
        text, scores = recognize_positions(binary, self.model, self.metadata)
        confidence = float(np.mean(scores))
        reason = "low_confidence" if confidence < CONFIDENCE_THRESHOLD else None
        return PerforationResult(text, scores, confidence, reason)
