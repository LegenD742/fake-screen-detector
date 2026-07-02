"""
predict.py -- real photo vs. screen-recapture detector.

Usage:
    python predict.py some_image.jpg
    -> prints a single float in [0, 1]: 0 = real photo, 1 = photo of a screen.

No trained deep model. Combines three hand-engineered, fast signals:
  - moire_score      (FFT periodic pixel-grid interference)
  - highlight_score  (compact specular glare typical of glossy flat screens)
  - banding_score    (exact-zero-gradient quantization in smooth regions)

Weights and threshold in WEIGHTS/THRESHOLD below are placeholders until
calibrated on real labeled data via calibrate.py -- see calibrate.py output.
"""

import sys
from features import extract_features

# Default weights -- overwritten by calibrate.py once real data is available.
WEIGHTS = {"moire": 0.5, "highlight": 0.3, "banding": 0.2}
BIAS = 0.0


def predict(path):
    feats = extract_features(path)
    score = sum(WEIGHTS[k] * feats[k] for k in WEIGHTS) + BIAS
    return max(0.0, min(1.0, score))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    print(f"{predict(sys.argv[1]):.4f}")