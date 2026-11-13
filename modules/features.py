"""
features.py — 33-dimensional thermal feature extractor for solar module images.

Each 24×40 grayscale image is mapped to a 33-dim float32 vector capturing:

  Group 1 — Global Intensity Statistics  (8 features)
    Mean, Median, Std, Variance, Min, Max, Range, Skewness

  Group 2 — Quantiles & Spread           (5 features)
    P10, P25, P75, P90, IQR

  Group 3 — Thermal Contrast             (4 features)
    Max/Mean ratio, Min/Mean ratio, Top-5% mean, Bottom-5% mean

  Group 4 — Spatial Gradients (Sobel)    (6 features)
    Mean |Gx|, Std |Gx|, Mean |Gy|, Std |Gy|, Mean |G|, Max |G|

  Group 5 — Heat Distribution            (4 features)
    Normalized centroid X, centroid Y, spread X, spread Y

  Group 6 — Texture & Anomaly Area       (6 features)
    Shannon entropy, Energy, Laplacian variance,
    Hot-area ratio (>μ+2σ), Cold-area ratio (<μ−2σ), CoV

Total: 8 + 5 + 4 + 6 + 4 + 6 = 33

Complexity per image: O(H·W) = O(960) → < 1 ms/image on CPU.
Batch extraction of 20,000 images takes roughly 30–60 s.
"""

import numpy as np
from scipy import ndimage
from scipy.stats import skew, entropy as scipy_entropy
from PIL import Image
from tqdm import tqdm

# Ordered names — used for axis labels in plots
FEATURE_NAMES = [
    # Group 1: Global Intensity
    "mean", "median", "std", "variance", "min", "max", "range", "skewness",
    # Group 2: Quantiles
    "p10", "p25", "p75", "p90", "iqr",
    # Group 3: Thermal Contrast
    "max_mean_ratio", "min_mean_ratio", "top5pct_mean", "bot5pct_mean",
    # Group 4: Gradients
    "gx_mean", "gx_std", "gy_mean", "gy_std", "gmag_mean", "gmag_max",
    # Group 5: Heat Distribution
    "centroid_x", "centroid_y", "spread_x", "spread_y",
    # Group 6: Texture & Area
    "entropy", "energy", "laplacian_var", "hot_ratio", "cold_ratio", "cov",
]
assert len(FEATURE_NAMES) == 33


def extract(img_arr: np.ndarray) -> np.ndarray:
    """Extract 33 thermal features from a 2-D float32 image array.

    Args:
        img_arr: np.ndarray of shape (H, W), values in [0, 255].

    Returns:
        np.ndarray of shape (33,), dtype float32.
    """
    img = img_arr.astype(np.float32)
    # Canonical shape: images are stored 40 tall × 24 wide (H=40, W=24)
    H, W = img.shape
    N    = H * W
    eps  = 1e-6

    # ── Group 1: Global Intensity Statistics ─────────────────────────────────
    mu      = float(img.mean())
    median  = float(np.median(img))
    sigma   = float(img.std())
    var     = float(img.var())
    mn      = float(img.min())
    mx      = float(img.max())
    rng     = mx - mn
    sk      = float(skew(img.ravel())) if sigma > eps else 0.0

    # ── Group 2: Quantiles & Spread ──────────────────────────────────────────
    p10, p25, p75, p90 = np.percentile(img, [10, 25, 75, 90])
    iqr = float(p75 - p25)

    # ── Group 3: Thermal Contrast ─────────────────────────────────────────────
    max_mean = mx / (mu + eps)
    min_mean = mn / (mu + eps)
    flat  = np.sort(img.ravel())
    k5    = max(1, int(N * 0.05))
    top5  = float(flat[-k5:].mean())
    bot5  = float(flat[:k5].mean())

    # ── Group 4: Spatial Gradients (Sobel) ────────────────────────────────────
    # Sobel returns values in range ≈ [-1020, 1020]; divide by 8 to normalise
    gx  = ndimage.sobel(img, axis=1) / 8.0
    gy  = ndimage.sobel(img, axis=0) / 8.0
    gm  = np.hypot(gx, gy)
    agx = np.abs(gx)
    agy = np.abs(gy)

    gx_mean = float(agx.mean());  gx_std  = float(agx.std())
    gy_mean = float(agy.mean());  gy_std  = float(agy.std())
    gm_mean = float(gm.mean());   gm_max  = float(gm.max())

    # ── Group 5: Heat Distribution ────────────────────────────────────────────
    # Weighted centroid and second moment along each axis
    total = float(img.sum()) + eps
    gy_idx, gx_idx = np.indices((H, W))
    cx = float((gx_idx * img).sum() / total / (W - 1))   # normalised 0–1
    cy = float((gy_idx * img).sum() / total / (H - 1))
    dx = gx_idx / (W - 1) - cx
    dy = gy_idx / (H - 1) - cy
    sx = float(np.sqrt((dx**2 * img).sum() / total))
    sy = float(np.sqrt((dy**2 * img).sum() / total))

    # ── Group 6: Texture & Anomaly Area ──────────────────────────────────────
    hist, _ = np.histogram(img, bins=32, range=(0, 256), density=True)
    h       = hist[hist > 0]
    ent     = float(scipy_entropy(h, base=2)) if len(h) > 0 else 0.0

    norm_img = img / (mx + eps)
    energy   = float((norm_img**2).mean())

    lap_var  = float(ndimage.laplace(img).var())
    hot_r    = float((img > mu + 2*sigma).sum() / N)
    cold_r   = float((img < mu - 2*sigma).sum() / N)
    cov      = sigma / (mu + eps)

    return np.array([
        mu, median, sigma, var, mn, mx, rng, sk,            # 8
        p10, p25, p75, p90, iqr,                            # 5
        max_mean, min_mean, top5, bot5,                     # 4
        gx_mean, gx_std, gy_mean, gy_std, gm_mean, gm_max, # 6
        cx, cy, sx, sy,                                     # 4
        ent, energy, lap_var, hot_r, cold_r, cov,           # 6
    ], dtype=np.float32)


def extract_from_path(path: str) -> np.ndarray:
    """Open an image file and extract features."""
    with Image.open(path) as im:
        arr = np.array(im.convert("L"), dtype=np.float32)
    return extract(arr)


def build_feature_matrix(df, out_path="outputs/features_33d.npy", show_progress=True):
    """Extract features for every image in a DataFrame and save to .npy.

    Args:
        df       : DataFrame with column 'image_filepath', sorted by id
        out_path : where to save the (N, 33) matrix
        show_progress : show tqdm bar

    Returns:
        np.ndarray of shape (N, 33)
    """
    import os
    if os.path.exists(out_path):
        print(f"Loading existing feature matrix from {out_path}")
        return np.load(out_path)

    paths  = df["image_filepath"].tolist()
    N      = len(paths)
    matrix = np.zeros((N, 33), dtype=np.float32)

    it = tqdm(paths, desc="Extracting features", unit="img") if show_progress else paths
    for i, p in enumerate(it):
        matrix[i] = extract_from_path(p)

    np.save(out_path, matrix)
    print(f"Saved feature matrix {matrix.shape} → {out_path}")
    return matrix

