"""
Sanity check with SYNTHETIC images only -- this does NOT calibrate the
real model. It just verifies the feature-extraction code runs correctly
and responds in the expected direction before real photos are available.

Run: python3 test_synthetic.py
"""

import numpy as np
from PIL import Image
import os
from features import extract_features

os.makedirs("synthetic", exist_ok=True)


def make_natural_like(path, size=400, seed=0):
    """Smooth natural-ish image: low-freq blobs + organic noise, no periodicity."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (size, size))
    # smooth it to emulate natural low-frequency structure
    from scipy.ndimage import gaussian_filter
    base = gaussian_filter(base, sigma=8)
    base = (base - base.min()) / (base.max() - base.min())
    # add fine organic noise (like sensor/texture noise, non-periodic)
    noise = rng.normal(0, 0.03, (size, size))
    img = np.clip(base + noise, 0, 1)
    rgb = np.stack([img, img * 0.9 + 0.05, img * 0.8 + 0.1], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def make_screen_like(path, size=400, seed=0, pixel_pitch=6):
    """Emulate a screen recapture: underlying image + periodic pixel-grid
    interference (moire) + a compact bright glare blob + mild banding."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (size, size))
    from scipy.ndimage import gaussian_filter
    base = gaussian_filter(base, sigma=8)
    base = (base - base.min()) / (base.max() - base.min())

    # periodic grid interference
    yy, xx = np.mgrid[0:size, 0:size]
    grid = 0.15 * (np.sin(2 * np.pi * xx / pixel_pitch) + np.sin(2 * np.pi * yy / pixel_pitch))
    img = np.clip(base + grid, 0, 1)

    # quantize to emulate banding
    img = np.round(img * 24) / 24.0

    rgb = np.stack([img, img * 0.9 + 0.05, img * 0.8 + 0.1], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)

    # compact bright glare blob
    cy, cx = size // 3, size // 2
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    glare = r2 < (size // 6) ** 2
    rgb[glare] = np.clip(rgb[glare].astype(np.int16) + 120, 0, 255).astype(np.uint8)

    Image.fromarray(rgb).save(path)


if __name__ == "__main__":
    print("Generating synthetic test images...")
    for i in range(5):
        make_natural_like(f"synthetic/real_{i}.png", seed=i)
        make_screen_like(f"synthetic/screen_{i}.png", seed=i)

    print(f"\n{'file':<20} {'moire':>8} {'highlight':>10} {'banding':>8}")
    print("-" * 50)
    for i in range(5):
        for kind in ["real", "screen"]:
            path = f"synthetic/{kind}_{i}.png"
            f = extract_features(path)
            print(f"{path:<20} {f['moire']:>8.3f} {f['highlight']:>10.3f} {f['banding']:>8.3f}")