import sys
from features import extract_features

COEF = {"banding": 0.507830,"pixelgrid" : -0.913214}
INTERCEPT = 0.341730
THRESHOLD = 0.514330


def predict(path):
    f = extract_features(path)
    z = sum(COEF[k] * f[k] for k in COEF) + INTERCEPT
    prob = 1.0 / (1.0 + pow(2.718281828, -z))  
    return prob


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    score = predict(sys.argv[1])
    label = "SCREEN" if score >= THRESHOLD else "REAL"
    print(f"{score:.4f}  ({label})")
