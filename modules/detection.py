"""Object detection utilities for simulated multi-module PV array scenes.

`InfraredSolarModules` provides 20,000 single-module cropped IR images with
image-level fault labels and no native bounding-box annotations. To satisfy
bounding-box object detection cleanly without fabricating annotations or
introducing cross-split data leakage, `CompositeSolarArrayDataset` constructs
320x320 multi-module PV array scenes strictly within each canonical split
(`train`, `val`, `test`) from `outputs/splits.csv`, recording exact ground-truth
`[x1, y1, x2, y2]` coordinates and 1-indexed fault labels (`1..12`, with `0`
reserved for Faster R-CNN background).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from .data import CLASS_NAMES, NUM_CLASSES

DETECTION_ARCH_VERSION = "fasterrcnn_mv3_large_320_fpn_v1"
DETECTION_PREPROCESSING_VERSION = "composite_320x320_norm01_v1"
DETECTION_CLASS_NAMES = ["__background__", *CLASS_NAMES]


def resolve_image_path(filepath: str, base_dir: Optional[Path | str] = None) -> Path:
    """Resolve a split CSV image path regardless of current working directory."""
    raw = Path(filepath)
    if raw.exists():
        return raw.resolve()

    candidates: List[Path] = []
    if base_dir is not None:
        base = Path(base_dir).resolve()
        candidates.extend([base / raw, base / "notebooks" / raw])

    project_root = Path(__file__).resolve().parent.parent
    candidates.extend([project_root / raw, project_root / "notebooks" / raw])

    for cand in candidates:
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        f"Could not resolve image path '{filepath}' from base_dir={base_dir}"
    )


def _load_crop_array(path: Path) -> np.ndarray:
    """Load an infrared crop as a uint8 2D grayscale array (H, W)."""
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def _sample_slot_boxes(
    rng: np.random.Generator,
    scene_size: int,
    n_modules: int,
    crop_hw_list: Sequence[Tuple[int, int]],
    scale_range: Tuple[float, float],
    min_gap: int = 8,
) -> List[Tuple[int, int, int, int]]:
    """Sample non-overlapping `[x1, y1, x2, y2]` placements inside `scene_size`."""
    # Try structured 3x3 jittered PV-string layout first, then random collision-free fallback.
    boxes: List[Tuple[int, int, int, int]] = []
    margin = 10
    grid_rows, grid_cols = 3, 3
    cell_h = (scene_size - 2 * margin) / grid_rows
    cell_w = (scene_size - 2 * margin) / grid_cols

    grid_indices = rng.permutation(grid_rows * grid_cols)[:n_modules]
    for slot_idx, crop_hw in zip(grid_indices, crop_hw_list):
        r, c = divmod(int(slot_idx), grid_cols)
        orig_h, orig_w = crop_hw
        scale = float(rng.uniform(scale_range[0], scale_range[1]))
        h = max(24, int(round(orig_h * scale)))
        w = max(20, int(round(orig_w * scale)))

        # Clamp box size inside one grid cell minus safety gap so cells stay disjoint.
        max_cell_h = int(cell_h - min_gap)
        max_cell_w = int(cell_w - min_gap)
        if h > max_cell_h or w > max_cell_w:
            shrink = min(max_cell_h / h, max_cell_w / w)
            h = max(20, int(h * shrink))
            w = max(16, int(w * shrink))

        cy0 = margin + r * cell_h
        cx0 = margin + c * cell_w
        y_slack = max(0, int(cell_h - h - min_gap // 2))
        x_slack = max(0, int(cell_w - w - min_gap // 2))
        y1 = int(cy0 + min_gap // 4 + (rng.integers(0, y_slack + 1) if y_slack > 0 else 0))
        x1 = int(cx0 + min_gap // 4 + (rng.integers(0, x_slack + 1) if x_slack > 0 else 0))
        x2 = min(scene_size - 4, x1 + w)
        y2 = min(scene_size - 4, y1 + h)
        boxes.append((x1, y1, x2, y2))

    return boxes


class CompositeSolarArrayDataset(Dataset):
    """Multi-module 320x320 PV array detection dataset built strictly from one split.

    Every scene composites `min_modules..max_modules` crops sampled exclusively from
    the provided split (`train`, `val`, or `test`) onto a simulated ambient PV array
    background, returning `(image_tensor_3hw, target_dict)` compatible with
    `torchvision.models.detection.FasterRCNN`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        num_scenes: int,
        scene_size: int = 320,
        min_modules: int = 4,
        max_modules: int = 8,
        scale_range: Tuple[float, float] = (1.35, 1.95),
        seed: int = 42,
        class_balanced: bool = True,
        augment: bool = False,
        cache_scenes: bool = True,
        base_dir: Optional[Path | str] = None,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split '{split}'; expected train/val/test")

        sub = df[df["split"] == split].reset_index(drop=True)
        if sub.empty:
            raise ValueError(f"No rows found for split='{split}'")

        self.split = split
        self.num_scenes = int(num_scenes)
        self.scene_size = int(scene_size)
        self.min_modules = int(min_modules)
        self.max_modules = min(9, int(max_modules))
        self.scale_range = scale_range
        self.seed = int(seed)
        self.epoch = 0
        self.class_balanced = bool(class_balanced)
        self.augment = bool(augment)

        self.paths: List[Path] = [
            resolve_image_path(fp, base_dir=base_dir) for fp in sub["image_filepath"]
        ]
        self.labels0: np.ndarray = sub["label"].to_numpy(dtype=np.int64)

        # Group indices by 0-indexed class for balanced scene sampling.
        self._class_indices: Dict[int, np.ndarray] = {
            c: np.where(self.labels0 == c)[0] for c in range(NUM_CLASSES)
        }
        self._present_classes = [c for c, idxs in self._class_indices.items() if len(idxs) > 0]

        # Preload uint8 grayscale crops (~20 MB total across 20k 40x24 images) for fast compositing.
        self._crops: List[np.ndarray] = [_load_crop_array(p) for p in self.paths]

        self._cached: Optional[List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]] = None
        if cache_scenes:
            self._cached = [self._build_scene(idx, epoch=0) for idx in range(self.num_scenes)]

    def set_epoch(self, epoch: int) -> None:
        """Advance the deterministic scene generator epoch for dynamic training."""
        self.epoch = int(epoch)

    def _sample_crop_indices(self, rng: np.random.Generator, count: int) -> np.ndarray:
        if not self.class_balanced:
            return rng.integers(0, len(self.paths), size=count)

        chosen_classes = rng.choice(self._present_classes, size=count, replace=True)
        return np.array(
            [rng.choice(self._class_indices[int(c)]) for c in chosen_classes],
            dtype=np.int64,
        )

    def _build_scene(
        self, idx: int, epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Seed deterministically per (base_seed, epoch, idx) so DataLoader order never leaks RNG state.
        scene_seed = (self.seed * 1_000_003 + epoch * 100_003 + idx) & 0xFFFFFFFF
        rng = np.random.default_rng(scene_seed)

        # Simulated ambient PV farm thermal canvas: smooth slight gradient + low-level thermal noise + mounting rails.
        base_temp = float(rng.uniform(22.0, 44.0))
        gy = np.linspace(-3.5, 3.5, self.scene_size, dtype=np.float32)[:, None]
        gx = np.linspace(-2.5, 2.5, self.scene_size, dtype=np.float32)[None, :]
        canvas = (
            base_temp
            + gy * float(rng.uniform(-1.0, 1.0))
            + gx * float(rng.uniform(-1.0, 1.0))
            + rng.normal(0.0, 5.5, size=(self.scene_size, self.scene_size)).astype(np.float32)
        )
        # Subtle horizontal PV rack rails matching 3-row array layout.
        for rail_y in (106, 213):
            y_jit = rail_y + int(rng.integers(-4, 5))
            canvas[max(0, y_jit - 1) : min(self.scene_size, y_jit + 2), :] += float(
                rng.uniform(3.0, 7.0)
            )

        n_modules = int(rng.integers(self.min_modules, self.max_modules + 1))
        crop_indices = self._sample_crop_indices(rng, n_modules)
        crop_hw_list = [self._crops[int(ci)].shape[:2] for ci in crop_indices]
        boxes = _sample_slot_boxes(
            rng,
            scene_size=self.scene_size,
            n_modules=n_modules,
            crop_hw_list=crop_hw_list,
            scale_range=self.scale_range,
        )

        labels_list: List[int] = []
        valid_boxes: List[Tuple[int, int, int, int]] = []
        for ci, (x1, y1, x2, y2) in zip(crop_indices, boxes):
            crop = self._crops[int(ci)].copy()
            if self.augment:
                if rng.random() < 0.5:
                    crop = np.fliplr(crop)
                if rng.random() < 0.5:
                    crop = np.flipud(crop)

            tw, th = x2 - x1, y2 - y1
            if tw <= 2 or th <= 2:
                continue

            resized = np.asarray(
                Image.fromarray(crop, mode="L").resize((tw, th), resample=Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            if self.augment:
                gain = float(rng.uniform(0.92, 1.08))
                bias = float(rng.uniform(-6.0, 6.0))
                resized = resized * gain + bias

            canvas[y1:y2, x1:x2] = np.clip(resized, 0.0, 255.0)
            valid_boxes.append((x1, y1, x2, y2))
            # Shift class labels by +1 because Faster R-CNN reserves label 0 for background.
            labels_list.append(int(self.labels0[int(ci)]) + 1)

        canvas = np.clip(canvas, 0.0, 255.0) / 255.0
        img_3ch = np.repeat(canvas[None, :, :], 3, axis=0).astype(np.float32)

        boxes_t = torch.tensor(valid_boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels_list, dtype=torch.int64)
        area_t = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area_t,
            "iscrowd": torch.zeros((len(valid_boxes),), dtype=torch.int64),
        }
        return torch.from_numpy(img_3ch), target

    def __len__(self) -> int:
        return self.num_scenes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if self._cached is not None:
            return self._cached[idx]
        return self._build_scene(idx, epoch=self.epoch)


def detection_collate_fn(
    batch: Sequence[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Collate variable-box-count detection samples for torchvision Faster R-CNN."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_fasterrcnn(
    num_classes: int = NUM_CLASSES + 1,
    pretrained: bool = True,
    min_size: int = 320,
    max_size: int = 320,
    trainable_backbone_layers: int = 5,
    box_score_thresh: float = 0.05,
) -> torch.nn.Module:
    """Build `fasterrcnn_mobilenet_v3_large_320_fpn` with a `num_classes` FastRCNNPredictor."""
    weights = (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
    )
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights,
        min_size=min_size,
        max_size=max_size,
        trainable_backbone_layers=trainable_backbone_layers,
        box_score_thresh=box_score_thresh,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute exact pairwise IoU matrix of shape `[N, M]` for `[x1, y1, x2, y2]` boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)

    b1 = np.asarray(boxes1, dtype=np.float64)
    b2 = np.asarray(boxes2, dtype=np.float64)

    x1 = np.maximum(b1[:, None, 0], b2[None, :, 0])
    y1 = np.maximum(b1[:, None, 1], b2[None, :, 1])
    x2 = np.minimum(b1[:, None, 2], b2[None, :, 2])
    y2 = np.minimum(b1[:, None, 3], b2[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = np.maximum(0.0, b1[:, 2] - b1[:, 0]) * np.maximum(0.0, b1[:, 3] - b1[:, 1])
    area2 = np.maximum(0.0, b2[:, 2] - b2[:, 0]) * np.maximum(0.0, b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)


def _compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute all-point interpolated Precision-Recall AUC (Pascal VOC 2010+ standard)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _to_numpy(val: Any) -> np.ndarray:
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    return np.asarray(val)


def compute_detection_metrics(
    predictions: Sequence[Dict[str, Any]],
    targets: Sequence[Dict[str, Any]],
    iou_thresholds: Tuple[float, ...] = (0.50, 0.75),
    score_threshold: float = 0.50,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, Any]:
    """Compute exact per-class AP, macro mAP, precision, recall, and matched IoU.

    Labels are expected to be 1-indexed (`1..num_classes`) matching Faster R-CNN outputs.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(targets)} targets"
        )

    pred_np = [
        {
            "boxes": _to_numpy(p["boxes"]),
            "labels": _to_numpy(p["labels"]).astype(np.int64),
            "scores": _to_numpy(p["scores"]).astype(np.float64),
        }
        for p in predictions
    ]
    tgt_np = [
        {
            "boxes": _to_numpy(t["boxes"]),
            "labels": _to_numpy(t["labels"]).astype(np.int64),
        }
        for t in targets
    ]

    per_class_rows: List[Dict[str, Any]] = []
    # Track matched IoUs and operating-point P/R at the primary threshold (iou_thresholds[0]).
    primary_iou = float(iou_thresholds[0])
    total_tp_op = 0
    total_fp_op = 0
    total_gt_op = 0
    matched_ious_op: List[float] = []

    for cls_id in range(1, num_classes + 1):
        cls_name = CLASS_NAMES[cls_id - 1]
        gt_by_img: Dict[int, np.ndarray] = {}
        total_gt = 0
        for img_idx, tgt in enumerate(tgt_np):
            mask = tgt["labels"] == cls_id
            boxes = tgt["boxes"][mask]
            if len(boxes) > 0:
                gt_by_img[img_idx] = boxes
                total_gt += len(boxes)

        # Gather all predictions for this class sorted descending by confidence score.
        cls_preds: List[Tuple[int, float, np.ndarray]] = []
        for img_idx, pred in enumerate(pred_np):
            mask = pred["labels"] == cls_id
            for box, score in zip(pred["boxes"][mask], pred["scores"][mask]):
                cls_preds.append((img_idx, float(score), box))
        cls_preds.sort(key=lambda item: item[1], reverse=True)

        ap_by_thr: Dict[float, float] = {}
        cls_tp_op = 0
        cls_fp_op = 0

        for iou_thr in iou_thresholds:
            if total_gt == 0:
                ap_by_thr[iou_thr] = float("nan")
                continue

            used: Dict[int, np.ndarray] = {
                img_idx: np.zeros(len(boxes), dtype=bool)
                for img_idx, boxes in gt_by_img.items()
            }
            tp = np.zeros(len(cls_preds), dtype=np.float64)
            fp = np.zeros(len(cls_preds), dtype=np.float64)

            for p_idx, (img_idx, score, p_box) in enumerate(cls_preds):
                gt_boxes = gt_by_img.get(img_idx)
                if gt_boxes is None or len(gt_boxes) == 0:
                    fp[p_idx] = 1.0
                    if abs(iou_thr - primary_iou) < 1e-9 and score >= score_threshold:
                        cls_fp_op += 1
                    continue

                ious = box_iou_matrix(p_box[None, :], gt_boxes)[0]
                best_gt = int(np.argmax(ious))
                best_iou = float(ious[best_gt])

                if best_iou >= iou_thr and not used[img_idx][best_gt]:
                    tp[p_idx] = 1.0
                    used[img_idx][best_gt] = True
                    if abs(iou_thr - primary_iou) < 1e-9 and score >= score_threshold:
                        cls_tp_op += 1
                        matched_ious_op.append(best_iou)
                else:
                    fp[p_idx] = 1.0
                    if abs(iou_thr - primary_iou) < 1e-9 and score >= score_threshold:
                        cls_fp_op += 1

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / max(total_gt, 1)
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
            ap_by_thr[iou_thr] = _compute_ap_from_pr(recalls, precisions)

        prec_op = cls_tp_op / max(cls_tp_op + cls_fp_op, 1)
        rec_op = cls_tp_op / max(total_gt, 1)
        total_tp_op += cls_tp_op
        total_fp_op += cls_fp_op
        total_gt_op += total_gt

        row: Dict[str, Any] = {
            "class_id": cls_id,
            "class_name": cls_name,
            "gt_count": int(total_gt),
            "pred_count_thr": int(cls_tp_op + cls_fp_op),
            "tp_thr": int(cls_tp_op),
            "fp_thr": int(cls_fp_op),
            f"precision@{score_threshold:.2f}": float(prec_op),
            f"recall@{score_threshold:.2f}": float(rec_op),
        }
        for iou_thr in iou_thresholds:
            row[f"AP@{iou_thr:.2f}"] = float(ap_by_thr[iou_thr])
        per_class_rows.append(row)

    per_class_df = pd.DataFrame(per_class_rows)
    summary: Dict[str, Any] = {
        "score_threshold": float(score_threshold),
        "total_gt": int(total_gt_op),
        "total_tp": int(total_tp_op),
        "total_fp": int(total_fp_op),
        "micro_precision": float(total_tp_op / max(total_tp_op + total_fp_op, 1)),
        "micro_recall": float(total_tp_op / max(total_gt_op, 1)),
        "mean_matched_iou": float(np.mean(matched_ious_op)) if matched_ious_op else 0.0,
        "per_class_df": per_class_df,
    }
    prec_m = summary["micro_precision"]
    rec_m = summary["micro_recall"]
    summary["micro_f1"] = (
        float(2 * prec_m * rec_m / (prec_m + rec_m)) if (prec_m + rec_m) > 0 else 0.0
    )
    for iou_thr in iou_thresholds:
        col = f"AP@{iou_thr:.2f}"
        summary[f"m{col}"] = float(per_class_df[col].dropna().mean())
    return summary

