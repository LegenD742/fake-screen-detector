"""
diagnose.py -- run this LOCALLY, in the same folder as calibrate.py.
Shows exactly which photos are misclassified, using cross-validation so
EVERY photo gets an honest out-of-sample prediction (not just a random 20%).

Run:
    python3 diagnose.py
"""

import os
import glob
import numpy as np
from features import extract_features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

REAL_DIR = "data/real"
SCREEN_DIR = "data/screen"


def main():
    real_paths = sorted(glob.glob(os.path.join(REAL_DIR, "*")))
    screen_paths = sorted(glob.glob(os.path.join(SCREEN_DIR, "*")))

    # figure out feature names dynamically from whatever features.py returns
    sample_feats = extract_features(real_paths[0])
    feat_names = sorted(sample_feats.keys())
    print(f"Using features: {feat_names}\n")

    X, y, paths = [], [], []
    for p in real_paths:
        f = extract_features(p)
        X.append([f[k] for k in feat_names])
        y.append(0)
        paths.append(p)
    for p in screen_paths:
        f = extract_features(p)
        X.append([f[k] for k in feat_names])
        y.append(1)
        paths.append(p)

    X = np.array(X)
    y = np.array(y)
    paths = np.array(paths)

    # 5-fold cross-validated predictions: every image is predicted by a
    # model that never saw it during training -- honest, and uses all
    # your data instead of wasting most of it on a single train split
    n_splits = min(5, min(np.bincount(y)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression()
    probs = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]

    preds = (probs >= 0.5).astype(int)
    correct = preds == y
    acc = correct.mean()
    print(f"Cross-validated accuracy across all {len(y)} photos: {acc*100:.1f}%\n")

    wrong_idx = np.where(~correct)[0]
    print(f"{len(wrong_idx)} misclassified photos:\n")
    print(f"{'file':<45} {'true':<8} {'pred':<8} {'score':<8}")
    print("-" * 75)
    # sort by how confidently wrong they were (most confident mistakes first --
    # these are the most informative to go look at)
    wrong_idx = sorted(wrong_idx, key=lambda i: abs(probs[i] - (1 - y[i])))
    for i in wrong_idx:
        true_label = "real" if y[i] == 0 else "screen"
        pred_label = "real" if preds[i] == 0 else "screen"
        print(f"{paths[i]:<45} {true_label:<8} {pred_label:<8} {probs[i]:<8.3f}")

    print()
    print("Also printing per-feature averages, correct vs incorrect,")
    print("to see which feature is least reliable:\n")
    for k in feat_names:
        col = X[:, feat_names.index(k)]
        print(f"  {k}: correct_mean={col[correct].mean():.3f}  wrong_mean={col[~correct].mean():.3f}")

    print()
    print("Go open the misclassified files listed above and look at them.")
    print("Common patterns to check for:")
    print(" - False positives (real->screen): glossy/reflective real objects,")
    print("   glass, polished metal, wet surfaces, real screens that are OFF")
    print(" - False negatives (screen->real): screen photographed from far")
    print("   away (grid too fine to alias), matte/e-ink displays, very")
    print("   bright/overexposed shots that wash out the pixel grid")


if __name__ == "__main__":
    main()