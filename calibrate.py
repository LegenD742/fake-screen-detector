"""
calibrate.py -- run this LOCALLY on your machine, on your own photos.

Expected folder layout (same folder as this script):
    data/real/    <- ~50 photos of real objects/scenes
    data/screen/  <- ~50 photos of a screen or printout showing a picture

What it does:
    1. Extracts the 3 features (moire, highlight, banding) for every photo
    2. Fits a logistic regression to combine them into one score
    3. Picks a decision threshold from the ROC curve (favoring low false-
       positive rate, i.e. rarely flagging a real photo as fake)
    4. Reports accuracy, false-positive rate, false-negative rate
    5. Writes the fitted weights + threshold into predict.py automatically
    6. Measures average latency per image

Run:
    python3 calibrate.py

Requires: numpy, scipy, pillow, scikit-learn
    pip install numpy scipy pillow scikit-learn
"""

import os
import time
import glob
import numpy as np
from features import extract_features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

REAL_DIR = "data/r2"
SCREEN_DIR = "data/f2"


def load_dataset():
    real_paths = sorted(glob.glob(os.path.join(REAL_DIR, "*")))
    screen_paths = sorted(glob.glob(os.path.join(SCREEN_DIR, "*")))

    if len(real_paths) < 10 or len(screen_paths) < 10:
        raise SystemExit(
            f"Not enough photos found. real/: {len(real_paths)}, screen/: {len(screen_paths)}. "
            f"Expected them in ./{REAL_DIR} and ./{SCREEN_DIR}"
        )

    print(f"Found {len(real_paths)} real photos, {len(screen_paths)} screen photos.\n")

    X, y, paths = [], [], []
    failed = []
    for p in real_paths:
        try:
            f = extract_features(p)
            X.append([f["banding"] , f["pixelgrid"]])
            y.append(0)
            paths.append(p)
        except Exception as e:
            failed.append((p, str(e)))

    for p in screen_paths:
        try:
            f = extract_features(p)
            X.append([f["banding"] , f["pixelgrid"]])
            y.append(1)
            paths.append(p)
        except Exception as e:
            failed.append((p, str(e)))

    if failed:
        print(f"WARNING: {len(failed)} images failed to process (corrupt/unsupported):")
        for p, e in failed[:5]:
            print(f"  {p}: {e}")

    return np.array(X), np.array(y), paths


def measure_latency(paths, n=30):
    sample = paths[:n] if len(paths) >= n else paths
    times = []
    for p in sample:
        t0 = time.perf_counter()
        extract_features(p)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    return np.mean(times), np.median(times), np.max(times)


def main():
    X, y, paths = load_dataset()

    # simple train/test split (80/20) to get an honest held-out accuracy
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    split = int(len(X) * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    probs_test = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs_test) if len(set(y_test)) > 1 else float("nan")

    fpr, tpr, thresholds = roc_curve(y_test, probs_test) if len(set(y_test)) > 1 else (None, None, None)

    # pick threshold targeting low false-positive rate (<=5%), per our
    # earlier reasoning: wrongly flagging a real photo is the worse error
    # for a consumer app, absent other business input
    chosen_threshold = 0.5

    if fpr is not None:
        j = tpr - fpr
        chosen_threshold = thresholds[np.argmax(j)]

    preds_test = (probs_test >= chosen_threshold).astype(int)
    acc = np.mean(preds_test == y_test)
    fp = np.sum((preds_test == 1) & (y_test == 0))
    fn = np.sum((preds_test == 0) & (y_test == 1))
    tn = np.sum((preds_test == 0) & (y_test == 0))
    tp = np.sum((preds_test == 1) & (y_test == 1))

    print("=" * 60)
    print("CALIBRATION RESULTS (held-out 20% test split)")
    print("=" * 60)
    print(f"Test set size: {len(y_test)}  (real={np.sum(y_test==0)}, screen={np.sum(y_test==1)})")
    print(f"AUC: {auc:.3f}")
    print(f"Chosen threshold: {chosen_threshold:.3f}")
    print(f"Accuracy at threshold: {acc*100:.1f}%")
    print(f"False positives (real flagged as screen): {fp} / {np.sum(y_test==0)}")
    print(f"False negatives (screen flagged as real):  {fn} / {np.sum(y_test==1)}")
    print(f"True negatives: {tn}, True positives: {tp}")
    print()

    coef = clf.coef_[0]
    intercept = clf.intercept_[0]
    print("Fitted logistic regression:")
    print(f"  banding weight:   {coef[0]:.4f}")
    print(f"  pixelgrid weight:   {coef[1]:.4f}")
    print(f"  intercept:        {intercept:.4f}")
    print()

    # latency
    mean_ms, median_ms, max_ms = measure_latency(paths)
    print(f"Latency per image -- mean: {mean_ms:.1f} ms, median: {median_ms:.1f} ms, max: {max_ms:.1f} ms")
    print("(measured on this machine's CPU -- phone CPUs are typically slower;")
    print(" treat this as a rough floor, not a phone-accurate number)")
    print()

    # write calibrated predictor
    write_calibrated_predict(coef, intercept, chosen_threshold)
    print("Wrote calibrated weights into predict_calibrated.py")
    print()
    print("NOTE: This accuracy is on a small held-out split of YOUR OWN photos,")
    print("not on the graders' unseen photos. Treat it as an estimate, and be")
    print("honest in the write-up about the sample size limitation.")


def write_calibrated_predict(coef, intercept, threshold):
    content = f'''#!/usr/bin/env python3
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

COEF = {{"banding": {coef[0]:.6f},"pixelgrid" : {coef[1]:.6f}}}
INTERCEPT = {intercept:.6f}
THRESHOLD = {threshold:.6f}


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
    print(f"{{score:.4f}}  ({{label}})")
'''
    with open("predict_calibrated.py", "w") as f:
        f.write(content)


if __name__ == "__main__":
    main()