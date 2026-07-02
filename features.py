import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import sobel

MAX_DIM = 512  


def load_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_DIM / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return gray, rgb


def pixel_grid_score(gray):

    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)

    vert = np.mean(np.abs(gx))
    hori = np.mean(np.abs(gy))

    fx = np.abs(np.fft.rfft(np.mean(np.abs(gx), axis=0)))
    fy = np.abs(np.fft.rfft(np.mean(np.abs(gy), axis=1)))

    fx[0] = 0
    fy[0] = 0

    peak = max(fx.max(), fy.max())
    avg = (fx.mean() + fy.mean()) / 2.0 + 1e-6

    score = peak / avg

    return float(np.clip((score - 5.0) / 15.0, 0.0, 1.0))

def highlight_score(rgb):
    
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32)
    h, w = gray.shape

    thresh = np.percentile(gray, 98)
    bright = gray >= max(thresh, 235)  

    frac_bright = bright.mean()
    if frac_bright < 0.001:
        return 0.0  

    ys, xs = np.where(bright)
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    compactness = bright.sum() / max(bbox_area, 1)  

    score = compactness * min(1.0, frac_bright * 15.0)
    return float(np.clip(score, 0.0, 1.0))


def banding_score(rgb):
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32)

    blurred = np.asarray(Image.fromarray(gray.astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), dtype=np.float32)
    local_var = (gray - blurred) ** 2
    smooth_mask = local_var < np.percentile(local_var, 50)

    if smooth_mask.sum() < 200:
        return 0.0

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)

    gx_mask = smooth_mask[:, :-1] & smooth_mask[:, 1:]
    gy_mask = smooth_mask[:-1, :] & smooth_mask[1:, :]

    all_diffs = np.concatenate([gx[gx_mask], gy[gy_mask]])
    if all_diffs.size < 100:
        return 0.0

    exact_zero_frac = float(np.mean(all_diffs == 0))
    return float(np.clip((exact_zero_frac - 0.03) / 0.25, 0.0, 1.0))


def extract_features(path):
    gray, rgb = load_image(path)
    return {
        "banding": banding_score(rgb),
        "pixelgrid" : pixel_grid_score(gray),
    }