# eval_cnn.py
"""
Evaluate a trained garbage-classification model on the test split.

Outputs
- Console: overall acc, top-5 acc, per-class precision/recall/F1
- Files in --outdir:
  - confusion_matrix.png (normalized)
  - classification_report.txt
  - preds.npy (y_true, y_pred, probs)
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE

def read_class_names(path: Path) -> List[str]:
    """Read class names, one per line."""
    with path.open("r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def build_test_ds(test_dir: Path, img_size: Tuple[int, int], batch: int) -> tuple[tf.data.Dataset, List[str]]:
    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=batch, label_mode="int", shuffle=False
    )
    class_names = ds.class_names  # capture BEFORE wrapping
    ds = ds.cache().prefetch(tf.data.AUTOTUNE)
    return ds, class_names


def collect_labels_and_probs(model: tf.keras.Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run inference over ds; return y_true (N,) and probs (N,C)."""
    y_true_list, probs_list = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        y_true_list.append(y.numpy())
        probs_list.append(p)
    y_true = np.concatenate(y_true_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    return y_true, probs

def topk_acc(y_true: np.ndarray, probs: np.ndarray, k: int = 5) -> float:
    """Compute top-k accuracy for integer labels and softmax probs."""
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([yt in row for yt, row in zip(y_true, topk)]))

def plot_confusion(cm: np.ndarray, class_names: List[str], out_png: Path) -> None:
    """Save a normalized confusion matrix heatmap."""
    fig = plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (row-normalized)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data_splits"))
    ap.add_argument("--img-size", type=int, nargs=2, default=(192, 192))
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--class-names", type=Path, default=Path("class_names_saved.txt"))
    ap.add_argument("--outdir", type=Path, default=Path("eval_out"))
    args = ap.parse_args()

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)

    # Load model and class names
    model = tf.keras.models.load_model(args.model)
    cls_from_file = read_class_names(args.class_names)

    # Build test ds
    test_dir = args.data_root / "test"
    test_ds, cls_from_ds = build_test_ds(test_dir, tuple(args.img_size), args.batch)

    # Basic sanity: class order should match
    if cls_from_ds != cls_from_file:
        print("[warn] class_names file differs from directory scanning order.")
        print("Using directory order for metrics:", cls_from_ds)

    # Inference
    y_true, probs = collect_labels_and_probs(model, test_ds)
    y_pred = probs.argmax(axis=1)

    # Overall metrics
    acc = float(np.mean(y_pred == y_true))
    acc_top5 = topk_acc(y_true, probs, k=5)
    print(f"Accuracy: {acc:.4f}  Top-5: {acc_top5:.4f}")

    # Per-class report
    report = classification_report(y_true, y_pred, target_names=cls_from_ds, digits=4)
    (outdir / "classification_report.txt").write_text(report, encoding="utf-8")
    print("\nPer-class metrics:\n")
    print(report)

    # Confusion matrix (row-normalized)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(cls_from_ds))))
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    np.save(outdir / "preds.npy", {"y_true": y_true, "y_pred": y_pred, "probs": probs, "classes": cls_from_ds})
    plot_confusion(cm_norm, cls_from_ds, outdir / "confusion_matrix.png")
    print(f"Saved: {outdir/'confusion_matrix.png'}, {outdir/'classification_report.txt'}, {outdir/'preds.npy'}")

if __name__ == "__main__":
    main()
