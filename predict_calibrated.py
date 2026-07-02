#!/usr/bin/env python3
"""
predict_calibrated.py -- real photo vs. screen-recapture detector.
Weights fitted by calibrate.py on local labeled data.

Usage:
    python predict_calibrated.py some_image.jpg
    -> prints a float in [0, 1]: 0 = real photo, 1 = photo of a screen.
    -> also prints REAL or SCREEN based on the calibrated threshold.
"""

import sys
from features import extract_features

COEF = {"banding": 0.469147,"pixelgrid" : -0.875906}
INTERCEPT = 0.332748
THRESHOLD = 0.488845


def predict(path):
    f = extract_features(path)
    z = sum(COEF[k] * f[k] for k in COEF) + INTERCEPT
    prob = 1.0 / (1.0 + pow(2.718281828, -z))  # sigmoid
    return prob


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_calibrated.py <image_path>")
        sys.exit(1)
    score = predict(sys.argv[1])
    label = "SCREEN" if score >= THRESHOLD else "REAL"
    print(f"{score:.4f}  ({label})")
