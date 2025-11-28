"""Segmentation stage implementations (S0, S1)."""
from __future__ import annotations

import cv2
import numpy as np


class NoSegmentation:
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image


class HSVMaskSegmentation:
    def __init__(self, lower=(0, 50, 20), upper=(179, 255, 255)) -> None:
        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        masked = cv2.bitwise_and(image, image, mask=mask)
        return masked
