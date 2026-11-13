"""
data.py — Dataset utilities for Infrared Solar Module fault classification.

What this file provides:
  - CLASS_NAMES, CLASS_TO_IDX   : canonical label mapping (12 classes)
  - load_metadata_df()          : load module_metadata.json into a DataFrame
  - create_or_load_splits()     : stratified 80/10/10 split saved to splits.csv
  - compute_class_weights()     : 1/sqrt(freq) weights for imbalanced loss
  - SolarDataset                : PyTorch Dataset → (image, features, label)
  - get_dataloaders()           : builds train/val/test DataLoaders

Design note on leakage prevention:
  Feature standardization stats (feat_mean, feat_std) are computed ONLY from
  the training split, then applied to val and test. Single-channel images use
  fixed per-image scaling (/255) and constant mean/std (0.449, 0.226) for
  native-res models, or ImageNet RGB constants for pretrained backbones.
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

# ─── Label mapping ────────────────────────────────────────────────────────────
# Ordered alphabetically; No-Anomaly kept at index 0 (majority class)
CLASS_NAMES = [
    "No-Anomaly",    # 0 — 10 000 samples
    "Cell",          # 1 —  1 877
    "Cell-Multi",    # 2 —  1 288
    "Cracking",      # 3 —    940
    "Diode",         # 4 —  1 499
    "Diode-Multi",   # 5 —    175  ← rarest, only ~18 test images
    "Hot-Spot",      # 6 —    249
    "Hot-Spot-Multi",# 7 —    246
    "Offline-Module",# 8 —    827
    "Shadowing",     # 9 —  1 056
    "Soiling",       # 10 —   204
    "Vegetation",    # 11 —  1 639
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES   = len(CLASS_NAMES)


# ─── Metadata loader ──────────────────────────────────────────────────────────
def load_metadata_df(data_dir="data/InfraredSolarModules"):
    """Read module_metadata.json and return a tidy DataFrame.

    Columns: id (int), image_filepath (str), anomaly_class (str), label (int)
    """
    path = os.path.join(data_dir, "module_metadata.json")
    with open(path) as f:
        raw = json.load(f)

    rows = []
    for img_id, item in raw.items():
        cls = item["anomaly_class"]
        rows.append({
            "id":             int(img_id),
            "image_filepath": os.path.join(data_dir, item["image_filepath"]),
            "anomaly_class":  cls,
            "label":          CLASS_TO_IDX[cls],
        })

    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


# ─── Stratified split ─────────────────────────────────────────────────────────
def create_or_load_splits(data_dir="data/InfraredSolarModules",
                          splits_csv="outputs/splits.csv",
                          seed=42):
    """Return a DataFrame with a 'split' column ('train' | 'val' | 'test').

    If splits.csv already exists it is loaded directly (idempotent).
    Split ratios: 80 / 10 / 10 — stratified per class.
    IDs are verified to be disjoint across splits.
    """
    if os.path.exists(splits_csv):
        df = pd.read_csv(splits_csv)
        print(f"Loaded splits from {splits_csv}  "
              f"(train={( df.split=='train').sum()}  "
              f"val={(df.split=='val').sum()}  "
              f"test={(df.split=='test').sum()})")
        return df

    df = load_metadata_df(data_dir)
    print(f"Loaded {len(df)} samples, {NUM_CLASSES} classes.")

    # 80% train | 20% temp
    train, temp = train_test_split(df, test_size=0.20,
                                   random_state=seed, stratify=df["label"])
    # 10% val | 10% test  (50/50 of the 20% temp)
    val, test   = train_test_split(temp, test_size=0.50,
                                   random_state=seed, stratify=temp["label"])

    for part, name in [(train, "train"), (val, "val"), (test, "test")]:
        part = part.copy()
        part["split"] = name
        if name == "train":
            train = part
        elif name == "val":
            val = part
        else:
            test = part

    # Leakage check
    assert not (set(train.id) & set(val.id)),  "Train/val ID overlap!"
    assert not (set(train.id) & set(test.id)), "Train/test ID overlap!"
    assert not (set(val.id)   & set(test.id)), "Val/test ID overlap!"

    combined = pd.concat([train, val, test]).sort_values("id").reset_index(drop=True)
    combined.to_csv(splits_csv, index=False)
    print(f"Saved splits → {splits_csv}  "
          f"(train={len(train)}  val={len(val)}  test={len(test)})")
    return combined


# ─── Class weights for imbalanced loss ───────────────────────────────────────
def compute_class_weights(df_train):
    """1/sqrt(class_freq), normalized so mean weight = 1.

    Square-root dampening prevents the loss from being dominated by
    the rarest classes (Diode-Multi has only ~140 training images).

    Returns: torch.FloatTensor of shape (NUM_CLASSES,)
    """
    counts = df_train["label"].value_counts().sort_index().values.astype(float)
    w = 1.0 / np.sqrt(counts + 1e-6)
    w /= w.mean()
    return torch.tensor(w, dtype=torch.float32)


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────
class SolarDataset(Dataset):
    """Multimodal dataset: returns (image_tensor, feature_vector, label).

    image_tensor  : FloatTensor (C, H, W)  — normalized
    feature_vector: FloatTensor (33,)       — standardized with train stats
    label         : LongTensor scalar

    Args:
        df            : DataFrame slice for this split
        feature_matrix: np.ndarray (N, 33) aligned row-to-row with df
        feat_mean     : train-split feature means  (shape 33,)
        feat_std      : train-split feature stds   (shape 33,)
        img_mean      : scalar or list — pixel mean for normalization
        img_std       : scalar or list — pixel std  for normalization
        target_size   : (H, W) — (40, 24) for scratch ViT; (224, 224) for pretrained
        in_channels   : 1 (grayscale scratch) or 3 (repeat-to-RGB for pretrained)
        is_train      : enables random augmentation
    """
    def __init__(self, df, feature_matrix,
                 feat_mean, feat_std,
                 img_mean=0.449, img_std=0.226,
                 target_size=(40, 24), in_channels=1,
                 is_train=False):

        self.df             = df.reset_index(drop=True)
        self.feature_matrix = feature_matrix
        self.feat_mean      = feat_mean
        self.feat_std       = np.where(feat_std < 1e-6, 1.0, feat_std)
        self.target_size    = target_size
        self.in_channels    = in_channels

        # Build image transform pipeline
        ops = []
        if target_size != (40, 24):
            # Bilinear upsample for pretrained models (224×224)
            ops.append(T.Resize(target_size,
                                interpolation=T.InterpolationMode.BILINEAR,
                                antialias=True))
        if is_train:
            ops += [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                # Small translation (up to 2 px) via padding + random crop
                T.Pad(2),
                T.RandomCrop(target_size),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        ops.append(T.ToTensor())   # PIL [0,255] → Float [0,1]

        if in_channels == 3:
            # Normalize with ImageNet stats for 3-channel pretrained backbones
            ops.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]))
        else:
            ops.append(T.Normalize(mean=[img_mean], std=[img_std]))

        self.transform = T.Compose(ops)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_filepath"]
        
        # Robust path resolution whether called from root or notebooks/
        if not os.path.exists(img_path):
            if img_path.startswith("../") and os.path.exists(img_path[3:]):
                img_path = img_path[3:]
            elif not img_path.startswith("../") and os.path.exists(os.path.join("..", img_path)):
                img_path = os.path.join("..", img_path)

        # Image (convert directly to RGB for 3-channel pretrained models, or L for grayscale)
        mode = "RGB" if self.in_channels == 3 else "L"
        with Image.open(img_path) as pil:
            img_tensor = self.transform(pil.convert(mode))

        # Feature vector — standardized with train-split statistics
        raw  = self.feature_matrix[idx].astype(np.float32)
        feat = (raw - self.feat_mean) / self.feat_std
        feat_tensor = torch.tensor(feat, dtype=torch.float32)

        label = torch.tensor(row["label"], dtype=torch.long)
        return img_tensor, feat_tensor, label


# ─── DataLoader builder ───────────────────────────────────────────────────────
def get_dataloaders(splits_csv="outputs/splits.csv",
                    features_npy="outputs/features_33d.npy",
                    batch_size=64,
                    target_size=(40, 24),
                    in_channels=1,
                    num_workers=0):
    """Build train / val / test DataLoaders.

    Feature standardization stats (feat_mean, feat_std) are computed from train split only;
    image normalization uses fixed constants (0.449/0.226 for grayscale, ImageNet for RGB).

    Returns:
        train_loader, val_loader, test_loader  — DataLoader objects
        class_weights                          — FloatTensor (NUM_CLASSES,)
        feat_mean, feat_std                    — np arrays (33,), for saving
    """
    df_all    = pd.read_csv(splits_csv)
    feats_all = np.load(features_npy)           # shape (N, 33)

    # Align feature rows to df order (df is sorted by id)
    # features_33d.npy must be generated in the same id-sorted order
    is_train = (df_all["split"] == "train").values
    is_val   = (df_all["split"] == "val").values
    is_test  = (df_all["split"] == "test").values

    df_tr, df_va, df_te = df_all[is_train], df_all[is_val], df_all[is_test]
    ft_tr, ft_va, ft_te = feats_all[is_train], feats_all[is_val], feats_all[is_test]

    # Train-split statistics only
    feat_mean = ft_tr.mean(axis=0).astype(np.float32)
    feat_std  = ft_tr.std(axis=0).astype(np.float32)

    # Fixed grayscale normalization constants (retained across all seeds for checkpoint consistency)
    img_mean, img_std = 0.449, 0.226

    make_ds = lambda df, ft, train: SolarDataset(
        df, ft, feat_mean, feat_std,
        img_mean=img_mean, img_std=img_std,
        target_size=target_size, in_channels=in_channels,
        is_train=train
    )

    train_ds = make_ds(df_tr, ft_tr, True)
    val_ds   = make_ds(df_va, ft_va, False)
    test_ds  = make_ds(df_te, ft_te, False)

    use_pin = torch.cuda.is_available()
    kw = dict(num_workers=num_workers, pin_memory=use_pin)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    class_weights = compute_class_weights(df_tr)
    return train_loader, val_loader, test_loader, class_weights, feat_mean, feat_std


# ─── Split Version & Provenance Manifest ──────────────────────────────────────
SPLIT_VERSION = "v1_stratified_seed42"


def compute_file_sha256(path: str) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_or_verify_manifest(
    splits_csv: str = "outputs/splits.csv",
    features_npy: str = "outputs/features_33d.npy",
    manifest_path: str = "outputs/split_manifest.json",
    split_version: str = SPLIT_VERSION,
    strict: bool = True,
) -> dict:
    """Create or verify the dataset-split & preprocessing provenance manifest.

    Raises RuntimeError (if strict=True) when splits.csv or features_33d.npy
    differs from the recorded manifest without a matching split_version update,
    preventing silent reuse of stale checkpoints across modified splits.
    """
    df = pd.read_csv(splits_csv)
    splits_sha = compute_file_sha256(splits_csv)
    feats_sha  = compute_file_sha256(features_npy) if os.path.exists(features_npy) else None

    current = {
        "split_version": split_version,
        "split_seed": 42,
        "splits_sha256": splits_sha,
        "features_sha256": feats_sha,
        "counts": {
            "total": int(len(df)),
            "train": int((df["split"] == "train").sum()),
            "val":   int((df["split"] == "val").sum()),
            "test":  int((df["split"] == "test").sum()),
        },
        "preprocessing": {
            "grayscale_img_mean": 0.449,
            "grayscale_img_std":  0.226,
            "rgb_pretrained_mean": [0.485, 0.456, 0.406],
            "rgb_pretrained_std":  [0.229, 0.224, 0.225],
            "feature_standardization": "train_split_only",
        },
    }

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            recorded = json.load(f)
        if recorded.get("splits_sha256") != splits_sha or (
            feats_sha is not None and recorded.get("features_sha256") != feats_sha
        ):
            msg = (
                f"Split/feature checksum mismatch vs {manifest_path}! "
                f"Expected splits_sha256={recorded.get('splits_sha256')[:12]}..., "
                f"got {splits_sha[:12]}... Stale checkpoints must NOT be reused."
            )
            if strict:
                raise RuntimeError(msg)
            print("WARNING:", msg)
        return recorded

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(current, f, indent=2)
    return current

