"""
Dataset preparation utility for image classification tasks.

Validates images organized under class-named subfolders, performs a
deterministic stratified split into train/val/test, and materializes the split
files into an output directory. Optionally creates resized JPEG copies with a
center-crop that preserves the requested aspect ratio. Emits:

- class_names.txt           One class name per line (sorted)
- manifest_{split}.csv      CSV with columns: filepath,label (paths are relative
                            to the output root; forward slashes)
- stats.json                JSON summary including per-split counts, totals,
                            ratios, seed, and configuration used

Typical usage:

    python dataset_prep.py --input data_raw --output data_splits \
        --train 0.7 --val 0.15 --test 0.15 --seed 42

    python dataset_prep.py \
        --input data_raw \
        --output data_splits \
        --train 0.7 --val 0.15 --test 0.15 \
        --seed 42 \
        --link-mode copy \
        --resize 224 224

Notes:
- Class discovery is based on immediate subdirectories of the input root.
- Only files with extensions in ALLOWED_EXTS are considered.
- Image verification opens headers and drops unreadable/corrupted files.
- If --resize is provided, images are re-saved as high-quality JPEG with EXIF
  orientation applied and LANCZOS resampling.
- Hardlink/symlink failures fall back to copy with a warning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


# =====================
# Data model
# =====================
@dataclass
class Sample:
    """
    Container for a single image sample.

    Attributes:
        path: Absolute filesystem path to the source image file.
        label: Class label derived from the class subdirectory name.
    """
    path: Path
    label: str


# =====================
# Discovery utilities
# =====================
def _is_image_file(p: Path) -> bool:
    """Return True if path is a regular file with an allowed image extension."""
    return p.is_file() and p.suffix.lower() in ALLOWED_EXTS


def _iter_image_files(dir_path: Path) -> Iterable[Path]:
    """Yield image files directly under dir_path (non-recursive)."""
    for p in dir_path.iterdir():
        if _is_image_file(p):
            yield p


def discover_classes(root: Path) -> List[str]:
    """
    Discover class names under the input root.

    A class is any immediate subdirectory of ``root`` that contains at least
    one file with an allowed image extension. The returned list is sorted to
    enforce deterministic behavior across runs and platforms.

    Args:
        root: Directory containing class-named subdirectories.

    Returns:
        Sorted list of discovered class names.

    Raises:
        ValueError: If no class folders with images are found.
    """
    classes: List[str] = []
    for p in sorted(x for x in root.iterdir() if x.is_dir()):
        if any(True for _ in _iter_image_files(p)):
            classes.append(p.name)
    if not classes:
        raise ValueError(f"No class folders with images found under: {root}")
    return classes


def collect_samples(root: Path, classes: Sequence[str], max_per_class: int | None) -> List[Sample]:
    """
    Collect image samples from class subdirectories.

    Args:
        root: Input dataset root containing class subdirectories.
        classes: Iterable of class names to collect from.
        max_per_class: Optional cap on number of samples per class; if ``None``,
            all eligible files are collected.

    Returns:
        List of ``Sample`` instances for all collected files.

    Raises:
        ValueError: If no images are found across the specified classes.
    """
    samples: List[Sample] = []
    for cls in classes:
        cls_dir = root / cls
        imgs = sorted(_iter_image_files(cls_dir))
        if max_per_class is not None:
            imgs = imgs[:max_per_class]
        samples.extend(Sample(path=p.resolve(), label=cls) for p in imgs)
    if not samples:
        raise ValueError(f"No images found under: {root}")
    return samples


# =====================
# Validation / transforms
# =====================
def verify_image(path: Path) -> bool:
    """
    Perform a lightweight integrity check on an image file.

    Opens with Pillow and calls ``verify()`` to validate headers without
    decoding the full image.

    Args:
        path: Filesystem path to the image.

    Returns:
        True if the file looks like a valid image; False otherwise.
    """
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def center_crop_resize(im: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """
    Center-crop an image to match a target aspect ratio, then resize.

    EXIF orientation is applied. Downscale/upscale uses LANCZOS for quality.

    Args:
        im: Pillow image to transform.
        size: Tuple ``(width, height)`` specifying the output dimensions.

    Returns:
        A new Pillow image resized to ``size``.
    """
    target_w, target_h = size
    im = ImageOps.exif_transpose(im)  # respect camera orientation
    w, h = im.size
    aspect_target = target_w / target_h
    aspect = w / h
    if abs(aspect - aspect_target) > 1e-3:
        if aspect > aspect_target:
            new_w = int(h * aspect_target)
            left = (w - new_w) // 2
            crop = im.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / aspect_target)
            top = (h - new_h) // 2
            crop = im.crop((0, top, w, top + new_h))
    else:
        crop = im
    return crop.resize((target_w, target_h), Image.Resampling.LANCZOS)


# =====================
# Splitting
# =====================
def make_splits(
    samples: List[Sample],
    classes: Sequence[str],
    train_r: float,
    val_r: float,
    test_r: float,
    seed: int,
) -> Dict[str, List[Sample]]:
    """
    Create deterministic stratified train/val/test splits.

    Per-class shuffling is deterministic given ``seed``. Split sizes use
    index cuts to avoid ratio drift due to independent rounding.

    Args:
        samples: List of verified samples to split.
        classes: Ordered list of class names present in ``samples``.
        train_r: Ratio for the training split; must be non-negative.
        val_r: Ratio for the validation split; must be non-negative.
        test_r: Ratio for the test split; must be non-negative.
        seed: Random seed controlling the shuffling within each class.

    Returns:
        Mapping with keys ``{"train", "val", "test"}`` to lists of ``Sample``.

    Raises:
        ValueError: If the sum of ratios is not 1.0 (within tolerance).
    """
    if not math.isclose(train_r + val_r + test_r, 1.0, rel_tol=1e-6):
        raise ValueError("train + val + test ratios must sum to 1.0")

    rng = random.Random(seed)
    by_class: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by_class[s.label].append(s)

    for cls in classes:
        rng.shuffle(by_class[cls])

    splits = {"train": [], "val": [], "test": []}
    for cls in classes:
        items = by_class[cls]
        n = len(items)
        n_train = int(n * train_r)
        n_val = int(n * (train_r + val_r)) - n_train
        n_test = n - n_train - n_val

        # Minimal guard: keep at least 1 sample in train when possible
        if n >= 1 and n_train == 0:
            take = min(1, n)  # always 1 here
            n_train = take
            n_val = max(0, n - n_train - n_test)

        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train:n_train + n_val])
        splits["test"].extend(items[n_train + n_val:])
    return splits


# =====================
# Output materialization
# =====================
def prepare_out_dirs(out_root: Path, classes: Sequence[str]) -> None:
    """
    Create the output directory structure for all splits and classes.

    Args:
        out_root: Root directory where split folders are created.
        classes: Iterable of class names for which subdirectories are created
            under each split.
    """
    for split in ("train", "val", "test"):
        for cls in classes:
            (out_root / split / cls).mkdir(parents=True, exist_ok=True)


def _stable_name(src: Path, force_jpg: bool) -> str:
    """
    Create a stable filename based on source path, size, and mtime.

    Args:
        src: Source image path.
        force_jpg: If True, force .jpg extension (used when resizing).

    Returns:
        Deterministic filename string with short hash suffix.
    """
    st = src.stat()
    h = hashlib.sha1(
        f"{src.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")
    ).hexdigest()[:8]
    ext = ".jpg" if force_jpg else src.suffix.lower()
    return f"{src.stem}_{h}{ext}"


def place_file(
    src: Path,
    dst: Path,
    link_mode: str,
    resize_to: Tuple[int, int] | None,
) -> None:
    """
    Materialize a source image at the destination path.

    If ``resize_to`` is provided, the image is opened, converted to RGB,
    center-cropped to the target aspect ratio, resized, and saved as a JPEG with
    quality 95, regardless of original format. Otherwise, the file is copied or
    linked according to ``link_mode``. Link failures fall back to copy.

    Args:
        src: Source image path.
        dst: Destination path inside the output split directory.
        link_mode: One of ``{"copy", "hardlink", "symlink"}``.
        resize_to: Optional ``(width, height)`` to trigger resizing behavior.
    """
    if resize_to is not None:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im = center_crop_resize(im, resize_to)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="JPEG", quality=95)
        return

    try:
        if link_mode == "copy":
            shutil.copy2(src, dst)
        elif link_mode == "hardlink":
            os.link(src, dst)
        elif link_mode == "symlink":
            os.symlink(src, dst)
        else:
            raise ValueError(f"Unknown link_mode: {link_mode}")
    except OSError as e:
        print(f"[warn] {link_mode} failed for {src} → {dst} ({e}); falling back to copy")
        shutil.copy2(src, dst)


def write_manifest(csv_path: Path, rows: List[Tuple[str, str]]) -> None:
    """
    Write a CSV manifest for a dataset split.

    The file contains a header row ``filepath,label`` followed by relative paths
    (from the output root) and their corresponding class labels.

    Args:
        csv_path: Path to the manifest to write.
        rows: Sequence of ``(filepath, label)`` pairs.
    """
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label"])
        w.writerows(rows)


def write_class_names(path: Path, classes: Sequence[str]) -> None:
    """
    Persist class names to a text file, one per line.

    Args:
        path: Destination ``class_names.txt`` path.
        classes: Class names to write; order is preserved.
    """
    with path.open("w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")


def compute_stats(splits: Dict[str, List[Sample]]) -> Dict[str, Dict[str, int]]:
    """
    Compute per-split, per-class sample counts.

    Args:
        splits: Mapping of split name to list of ``Sample`` instances.

    Returns:
        Nested mapping ``{split: {class_name: count}}`` with classes sorted
        within each split.
    """
    out: Dict[str, Dict[str, int]] = {}
    for split, items in splits.items():
        c = Counter(s.label for s in items)
        out[split] = dict(sorted(c.items()))
    return out


# =====================
# CLI
# =====================
def main():
    """
    Command-line entry point for dataset preparation.

    Parses arguments, discovers classes, verifies images, performs stratified
    splitting, materializes files, writes manifests, and emits a JSON summary.
    Also prints a brief human-readable split summary to stdout.
    """
    p = argparse.ArgumentParser(description="Dataset preparation tool")
    p.add_argument("--input", type=Path, required=True, help="Root folder with class subfolders")
    p.add_argument("--output", type=Path, required=True, help="Output folder for splits")
    p.add_argument("--train", type=float, default=0.7, help="Train ratio")
    p.add_argument("--val", type=float, default=0.15, help="Validation ratio")
    p.add_argument("--test", type=float, default=0.15, help="Test ratio")
    p.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    p.add_argument("--max-per-class", type=int, default=None, help="Cap per-class samples")
    p.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="copy",
                   help="How to materialize files into split dirs")
    p.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=None,
                   help="If set, write resized JPEG copies at WxH")
    args = p.parse_args()

    in_root: Path = args.input
    out_root: Path = args.output
    out_root.mkdir(parents=True, exist_ok=True)

    classes = discover_classes(in_root)
    classes_sorted = classes  # already sorted in discover_classes

    # Pre-verify raw counts per class
    counts_pre_verify = {c: sum(1 for _ in _iter_image_files(in_root / c)) for c in classes_sorted}

    samples_all = collect_samples(in_root, classes_sorted, args.max_per_class)

    # Verify images
    good_samples: List[Sample] = []
    bad_count_by_class: Dict[str, int] = defaultdict(int)
    for s in samples_all:
        if verify_image(s.path):
            good_samples.append(s)
        else:
            bad_count_by_class[s.label] += 1

    if not good_samples:
        raise ValueError("All images failed verification. Check input directory.")

    total_bad = sum(bad_count_by_class.values())
    if total_bad > 0:
        print(f"[warn] Dropped {total_bad} corrupted or unreadable images.")
        for cls in sorted(bad_count_by_class):
            print(f"       {cls:<15} {bad_count_by_class[cls]}")

    splits = make_splits(
        samples=good_samples,
        classes=classes_sorted,
        train_r=args.train,
        val_r=args.val,
        test_r=args.test,
        seed=args.seed,
    )

    prepare_out_dirs(out_root, classes_sorted)

    manifests: Dict[str, List[Tuple[str, str]]] = {"train": [], "val": [], "test": []}
    force_jpg = args.resize is not None
    resize_tuple = tuple(args.resize) if args.resize else None

    for split, items in splits.items():
        for s in items:
            dst_name = _stable_name(s.path, force_jpg=force_jpg)
            dst_path = out_root / split / s.label / dst_name
            place_file(
                src=s.path,
                dst=dst_path,
                link_mode=args.link_mode,
                resize_to=resize_tuple,
            )
            rel = dst_path.relative_to(out_root).as_posix()
            manifests[split].append((rel, s.label))

    write_class_names(out_root / "class_names.txt", classes_sorted)
    for split, rows in manifests.items():
        write_manifest(out_root / f"manifest_{split}.csv", rows)

    stats_post = compute_stats(splits)
    animated_ext_present = any((in_root / c).glob("*.gif") or (in_root / c).glob("*.webp") for c in classes_sorted)

    stats_payload = {
        "class_names": classes_sorted,
        "counts_pre_verify": counts_pre_verify,
        "dropped_by_class": dict(bad_count_by_class),
        "counts": stats_post,
        "total": {k: sum(v.values()) for k, v in stats_post.items()},
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "seed": args.seed,
        "resize": args.resize,
        "link_mode": args.link_mode,
        "input_root": str(in_root),
        "output_root": str(out_root),
        "animated_ext_present": bool(animated_ext_present),  # GIF/WEBP present; first frame used if resized
    }
    with (out_root / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)

    print("\n=== Split summary ===")
    for split in ("train", "val", "test"):
        total = sum(stats_post[split].values())
        print(f"{split}: {total} images")
        for cls, n in stats_post[split].items():
            print(f"  {cls:<15} {n}")
    print("\nWrote:", out_root / "class_names.txt")
    print("Wrote:", out_root / "manifest_train.csv")
    print("Wrote:", out_root / "manifest_val.csv")
    print("Wrote:", out_root / "manifest_test.csv")
    print("Wrote:", out_root / "stats.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
