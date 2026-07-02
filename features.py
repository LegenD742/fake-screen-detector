import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import sobel

MAX_DIM = 512  # downsample target -- plenty of resolution for these signals, keeps it fast


def load_image(path):
    """Load image, return (gray float32 array, RGB uint8 array), both downsampled."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = MAX_DIM / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    return gray, rgb


# ---------------------------------------------------------------------------
# 1. Moire / periodic pixel-grid interference
# ---------------------------------------------------------------------------
def moire_score(gray):
    """
    Real-world images have a smooth, natural 1/f-like frequency spectrum.
    A photographed screen samples one pixel grid with another (the camera
    sensor), producing sharp, unnatural narrow-band peaks in the mid/high
    frequency range. We measure how "peaky" the spectrum is relative to its
    local neighborhood -- a natural image has no such spikes.
    """
    g = gray - gray.mean()
    # window to reduce edge artifacts in the FFT
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    g = g * win

    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F)
    mag = np.log1p(mag)

    h, w = mag.shape
    cy, cx = h // 2, w // 2

    # ignore the very low frequencies (DC / overall lighting gradients) --
    # radius mask
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / r.max()

    band_mask = (r_norm > 0.08) & (r_norm < 0.85)  # mid/high freq band, skip DC and extreme corners
    band = mag[band_mask]

    if band.size == 0:
        return 0.0

    median = np.median(band)
    mad = np.median(np.abs(band - median)) + 1e-6  # robust spread estimate

    # how many std-devs (robust) above local median is the peak?
    peak = band.max()
    peak_z = (peak - median) / mad

    # also measure how much energy sits in a narrow top percentile --
    # natural spectra decay smoothly, moire spectra have concentrated spikes
    top_frac = np.mean(band > (median + 6 * mad))

    # combine, then squash to roughly 0..1 with a soft cap
    raw = 0.15 * peak_z + 40.0 * top_frac
    return float(1.0 - np.exp(-raw / 8.0))


def pixel_grid_score(gray):
    """
    Detect repetitive vertical/horizontal pixel-grid energy that is common
    in photos of LCD/OLED screens.

    Returns:
        0 -> natural image
        1 -> strong pixel grid
    """

    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)

    # average edge strength
    vert = np.mean(np.abs(gx))
    hori = np.mean(np.abs(gy))

    # regularity via FFT of edge maps
    fx = np.abs(np.fft.rfft(np.mean(np.abs(gx), axis=0)))
    fy = np.abs(np.fft.rfft(np.mean(np.abs(gy), axis=1)))

    fx[0] = 0
    fy[0] = 0

    peak = max(fx.max(), fy.max())
    avg = (fx.mean() + fy.mean()) / 2.0 + 1e-6

    score = peak / avg

    return float(np.clip((score - 5.0) / 15.0, 0.0, 1.0))

# ---------------------------------------------------------------------------
# 2. Specular highlight compactness (screen glare)
# ---------------------------------------------------------------------------
def highlight_score(rgb):
    """
    Screens are flat and glossy: reflections/glare tend to form a single
    large, spatially compact, low-texture bright blob. Real 3D objects
    under normal lighting tend to have highlights that are smaller,
    more scattered, and follow object geometry.
    """
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32)
    h, w = gray.shape

    thresh = np.percentile(gray, 98)
    bright = gray >= max(thresh, 235)  # only count genuinely bright pixels

    frac_bright = bright.mean()
    if frac_bright < 0.001:
        return 0.0  # no meaningful highlight present

    # crude connected-component compactness via simple flood-fill-free proxy:
    # compare the bounding-box area of bright pixels to their actual count.
    ys, xs = np.where(bright)
    bbox_area = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
    compactness = bright.sum() / max(bbox_area, 1)  # 1.0 = perfectly fills its bbox (solid blob)

    # a single large solid blob (screen glare) -> high compactness AND high frac_bright
    # scattered small highlights (real scene) -> low compactness
    score = compactness * min(1.0, frac_bright * 15.0)
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 3. Color / gradient banding (display quantization)
# ---------------------------------------------------------------------------
def banding_score(rgb):
    """
    Real camera sensor noise means adjacent pixels in a smooth region
    (sky, wall, skin, out-of-focus background) are almost never *exactly*
    identical. Screen recaptures (quantized source + recompression) often
    contain literal flat runs -- exact-zero gradients -- in those same
    regions. We measure the fraction of exact-zero horizontal/vertical
    gradients within smooth areas as a direct quantization signal.
    """
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

    # natural sensor noise: this should be very low (typically < 2-5%).
    # heavily quantized/recompressed screen images: noticeably higher.
    return float(np.clip((exact_zero_frac - 0.03) / 0.25, 0.0, 1.0))


def extract_features(path):
    gray, rgb = load_image(path)
    return {
        "banding": banding_score(rgb),
        "pixelgrid" : pixel_grid_score(gray),
    }

