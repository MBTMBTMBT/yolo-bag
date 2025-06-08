#!/usr/bin/env python3
"""
Offline data-augmentation for a YOLOv11 dataset.
(same description as before)
"""

import argparse
import random
import uuid
from pathlib import Path
from typing import Optional          # ← added

import cv2
import numpy as np
import yaml
from tqdm import tqdm
import albumentations as A

# ─────────────────── mosaic shim ───────────────────
try:
    from albumentations.contrib.transforms import Mosaic as _Mosaic
    A.Mosaic = _Mosaic
except Exception:
    A.Mosaic = None
# ───────────────────────────────────────────────────

def _clip_boxes(bboxes, eps: float = 1e-6):
    fixed = []
    for x, y, w, h in bboxes:
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(max(w, eps), 1.0)
        h = min(max(h, eps), 1.0)
        if w > 0 and h > 0:
            fixed.append([x, y, w, h])
    return fixed

# ---------- augmentation pipelines (unchanged) ----------
def make_single_bag_aug():
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=900, p=1),
            A.RandomScale(scale_limit=(-0.4, 0.3), p=0.9),
            A.Affine(translate_percent=(-0.25, 0.25), rotate=(-15, 15), p=0.9),
            A.Perspective(scale=(0.05, 0.15), p=0.5),
            A.OneOf(
                [
                    A.RGBShift(15, 15, 15),
                    A.HueSaturationValue(15, 20, 15),
                    A.RandomBrightnessContrast(0.2, 0.2),
                ],
                p=0.7,
            ),
            A.GaussNoise((10, 60), p=0.5),
            A.Resize(480, 640, interpolation=cv2.INTER_AREA),
        ],
        bbox_params=A.BboxParams(format="yolo",
                                 min_visibility=0.2,
                                 label_fields=["class_labels"]),
    )

def make_mosaic_aug():
    if A.Mosaic is None:
        return None
    return A.Compose(
        [
            A.Mosaic(
                scale=(0.85, 1.15),
                additional_targets={f"image{i}": "image" for i in range(1, 4)},
                bbox_params=A.BboxParams(
                    format="yolo",
                    label_fields=[f"class_labels{i}" for i in range(4)],
                ),
            ),
            A.OneOf(
                [A.MotionBlur(3), A.GaussianBlur(3), A.MedianBlur(3)], p=0.3
            ),
            A.Resize(480, 640, interpolation=cv2.INTER_AREA),
        ]
    )

# ---------- helpers (unchanged) ----------
def load_yolo_label(path):
    bbs, cids = [], []
    with open(path) as f:
        for line in f:
            toks = line.strip().split()
            if not toks:
                continue
            cls, x, y, w, h = map(float, toks)
            cids.append(int(cls))
            bbs.append([x, y, w, h])
    return bbs, cids

def save_yolo_label(path, bbs, cids):
    with open(path, "w") as f:
        for bb, cid in zip(bbs, cids):
            f.write(f"{cid} {' '.join(f'{v:.6f}' for v in bb)}\n")

def random_bg(bg_dir: Path, hw):
    files = list(bg_dir.glob("*.*"))
    if not files:
        return None
    img = cv2.imread(str(random.choice(files)))
    return cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)

def paste_on_bg(fg_img, mask, bg_img):
    if bg_img is None:
        return fg_img
    mask3 = np.dstack([mask] * 3).astype(bool)
    out = bg_img.copy()
    out[mask3] = fg_img[mask3]
    return out

def augment_single(img, bbs, cids, aug, bg_dir):
    res = aug(image=img, bboxes=bbs, class_labels=cids)
    out_img, out_bbs, out_cls = res["image"], res["bboxes"], res["class_labels"]
    if bg_dir:
        mask = (cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY) > 0).astype("uint8")
        out_img = paste_on_bg(out_img, mask, random_bg(bg_dir, out_img.shape[:2]))
    return out_img, _clip_boxes(out_bbs), out_cls

# ---------- main split loop ----------
def augment_dataset(split_dir: Path, n_aug: int, bg_dir: Optional[Path]):   # ← fixed annotation
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    basic_aug = make_single_bag_aug()
    img_paths = list(img_dir.glob("*.*"))

    for img_p in tqdm(img_paths, desc=f"{split_dir.name:5} basic", ncols=80):
        img = cv2.imread(str(img_p))
        bbs, cids = load_yolo_label(lbl_dir / f"{img_p.stem}.txt")
        for _ in range(n_aug):
            new_img, new_bb, new_cls = augment_single(img.copy(), bbs, cids, basic_aug, bg_dir)
            if not new_bb:
                continue
            uid = uuid.uuid4().hex[:8]
            cv2.imwrite(str(img_dir / f"{img_p.stem}_aug_{uid}.jpg"), new_img)
            save_yolo_label(lbl_dir / f"{img_p.stem}_aug_{uid}.txt", new_bb, new_cls)

    mosaic_aug = make_mosaic_aug()
    if mosaic_aug is None:
        return
    random.shuffle(img_paths)
    for idx in range(0, len(img_paths), 4):
        if idx + 3 >= len(img_paths):
            break
        imgs, bbs, cls = [], [], []
        for p in img_paths[idx:idx + 4]:
            imgs.append(cv2.imread(str(p)))
            bb, cl = load_yolo_label(lbl_dir / f"{p.stem}.txt")
            bbs.append(bb)
            cls.append(cl)
        try:
            res = mosaic_aug(
                image=imgs[0], bboxes=bbs[0], class_labels=cls[0],
                **{f"image{i}": imgs[i] for i in range(1, 4)},
                **{f"bboxes{i}": bbs[i] for i in range(1, 4)},
                **{f"class_labels{i}": cls[i] for i in range(1, 4)},
            )
        except Exception:
            continue
        out_img, out_bb, out_cls = res["image"], _clip_boxes(res["bboxes"]), res["class_labels"]
        if not out_bb:
            continue
        uid = uuid.uuid4().hex[:8]
        cv2.imwrite(str(img_dir / f"mosaic_{uid}.jpg"), out_img)
        save_yolo_label(lbl_dir / f"mosaic_{uid}.txt", out_bb, out_cls)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Albumentations offline augmentation for YOLOv11")
    ap.add_argument("dataset_dir", help="processed_dataset root")
    ap.add_argument("--n_aug", type=int, default=3, help="copies per original image")
    ap.add_argument("--bg_dir", type=str, default=None, help="backgrounds folder (optional)")
    args = ap.parse_args()

    ds_root = Path(args.dataset_dir).expanduser().resolve()
    if not (ds_root / "train").exists():
        raise FileNotFoundError("train/ folder not found under dataset_dir")

    bg_dir = Path(args.bg_dir).expanduser().resolve() if args.bg_dir else None
    for split in ["train", "valid"]:
        augment_dataset(ds_root / split, args.n_aug, bg_dir)

    print(f"✔  Augmentation complete – new images live under {ds_root}")


if __name__ == "__main__":
    main()
