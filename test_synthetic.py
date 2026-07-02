import numpy as np
from PIL import Image
import os
from features import extract_features

os.makedirs("synthetic", exist_ok=True)


def make_natural_like(path, size=400, seed=0):
    
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (size, size))
    
    from scipy.ndimage import gaussian_filter
    base = gaussian_filter(base, sigma=8)
    base = (base - base.min()) / (base.max() - base.min())
    
    noise = rng.normal(0, 0.03, (size, size))
    img = np.clip(base + noise, 0, 1)
    rgb = np.stack([img, img * 0.9 + 0.05, img * 0.8 + 0.1], axis=-1)
    rgb = (rgb * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def make_screen_like(path, size=400, seed=0, pixel_pitch=6):
    
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, (size, size))
    from scipy.ndimage import gaussian_filter
    base = gaussian_filter(base, sigma=8)
    base = (base - base.min()) / (base.max() - base.min())

    yy, xx = np.mgrid[0:size, 0:size]
    grid = 0.15 * (np.sin(2 * np.pi * xx / pixel_pitch) + np.sin(2 * np.pi * yy / pixel_pitch))
    img = np.clip(base + grid, 0, 1)

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

    print(f"\n{'file':<20} {'pixelgrid':>8} {'banding':>8}")
    print("-" * 50)
    for i in range(5):
        for kind in ["real", "screen"]:
            path = f"synthetic/{kind}_{i}.png"
            f = extract_features(path)
            print(f"{path:<20} {f['pixelgrid']:>8.3f} {f['banding']:>8.3f}")