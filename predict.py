# predict.py -- real photo vs. screen-recapture detector.

# To use this file independentally :
#    python predict.py yourimgname.jpg
#    -> prints a single float in [0, 1]: 0 = real photo, 1 = photo of a screen.

# No trained deep model. Combines 2 hand engineered, fast signals:
#  - pixelgrid_score
#  - banding_score   


import sys
from features import extract_features

WEIGHTS = {"pixelgrid": 0.66, "banding": 0.34}
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