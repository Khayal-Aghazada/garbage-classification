"""
train_cnn2.py — Fast garbage-classification training with Keras.

Features
- MobileNetV2 transfer learning (frozen head -> fine-tune top layers)
- Mixed precision on GPU, XLA JIT (safe no-ops on CPU)
- Size/batch-tagged on-disk caching for tf.data pipelines
- Class weights to mitigate imbalance
- Smoke test that never touches caches
- Clean CLI for training and single-image prediction

Usage
-----
Train (fast baseline at 192px):
  python train_cnn2.py --data-root data_splits --img-size 192 192 --batch 96 \
    --freeze-epochs 1 --ft-epochs 6 --lr 1e-3 --ft-lr 5e-5 --unfreeze-layers 40 --model out_192.keras

Predict:
  python train_cnn2.py --predict "C:\\path\\to\\image.jpg" --model out_192.keras \
    --class-names data_splits/class_names.txt --img-size 192 192
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------
# Repro + hardware setup
# ---------------------------
def set_seed(seed: int) -> None:
    """Set seeds for basic reproducibility."""
    import random
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def setup_device() -> None:
    """Enable mixed precision + XLA on GPU; allow memory growth."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Memory growth
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        # Mixed precision
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass
        # XLA JIT (optional)
        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ---------------------------
# Data utils
# ---------------------------
def read_class_names(path: Path) -> List[str]:
    """Read class names file (one per line)."""
    with path.open("r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def count_files(p: Path) -> int:
    """Count regular files in a directory (non-recursive)."""
    return sum(1 for _ in p.iterdir() if _.is_file())


def compute_class_weights(train_dir: Path, class_names: List[str]) -> Dict[int, float]:
    """
    Inverse-frequency class weights.
    weight_c = N_total / (K * count_c)
    """
    counts = {c: count_files(train_dir / c) for c in class_names}
    total, k = sum(counts.values()), len(class_names)
    return {i: (total / (k * max(1, counts[c]))) for i, c in enumerate(class_names)}


def build_datasets(
    data_root: Path, img_size: Tuple[int, int], batch: int, cache_to_disk: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Build train/val/test datasets from folders. Cache to SSD tagged by size/batch.
    """
    train_dir, val_dir, test_dir = data_root / "train", data_root / "val", data_root / "test"
    train = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch, label_mode="int",
        shuffle=True, seed=42
    )
    class_names = train.class_names
    val = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=img_size, batch_size=batch, label_mode="int", shuffle=False
    )
    test = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=batch, label_mode="int", shuffle=False
    )

    if cache_to_disk:
        tag = f"{img_size[0]}x{img_size[1]}_b{batch}"
        train = train.cache((data_root / f"train_{tag}.cache").as_posix())
        val   = val.cache((data_root / f"val_{tag}.cache").as_posix())
        test  = test.cache((data_root / f"test_{tag}.cache").as_posix())
    else:
        val = val.cache()
        test = test.cache()

    train = train.prefetch(AUTOTUNE)
    val   = val.prefetch(AUTOTUNE)
    test  = test.prefetch(AUTOTUNE)
    return train, val, test, class_names


# ---------------------------
# Models
# ---------------------------
def build_mobilenetv2_head(num_classes: int, img_size: Tuple[int, int]) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    MobileNetV2 backbone (ImageNet weights) + GAP + Dropout + Dense softmax.
    Preprocessing is inside the graph.
    """
    H, W = img_size
    base = tf.keras.applications.MobileNetV2(
        input_shape=(H, W, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(H, W, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)  # expects raw 0..255
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    # Keep float32 logits under mixed precision
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs, name="garbage_mnv2")
    return model, base


def unfreeze_top_layers(base: tf.keras.Model, n_layers: int) -> None:
    """Unfreeze only the last n_layers of the backbone."""
    base.trainable = True
    for l in base.layers[:-n_layers]:
        l.trainable = False


# ---------------------------
# Smoke test (never cached)
# ---------------------------
def smoke_test(train_dir: Path, img_size: Tuple[int, int], batch: int) -> None:
    """
    Quick pipeline check:
    - Load 1 batch uncached at requested size
    - Build model and run a single train_on_batch
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=img_size, batch_size=batch, label_mode="int",
        shuffle=True, seed=42
    ).take(1)
    x, y = next(iter(ds))
    print(f"[Smoke] x={x.shape}, y={y.shape}")
    num_classes = int(y.numpy().max()) + 1
    model, _ = build_mobilenetv2_head(num_classes, img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    model.train_on_batch(x, y)
    print("[Smoke] OK")


# ---------------------------
# Training
# ---------------------------
def train(
    data_root: Path,
    img_size: Tuple[int, int],
    batch: int,
    freeze_epochs: int,
    ft_epochs: int,
    lr: float,
    ft_lr: float,
    unfreeze_layers: int,
    out_model_path: Path,
) -> None:
    # Smoke test before building cached datasets
    smoke_test(data_root / "train", img_size, batch)

    train_ds, val_ds, test_ds, class_names = build_datasets(
        data_root, img_size, batch, cache_to_disk=True
    )
    model, base = build_mobilenetv2_head(len(class_names), img_size)
    class_weights = compute_class_weights(data_root / "train", class_names)

    # Phase 1: train head only
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("out_best.keras", monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    if freeze_epochs > 0:
        print(f"[Train] Frozen base for {freeze_epochs} epoch(s)")
        model.fit(
            train_ds, validation_data=val_ds, epochs=freeze_epochs,
            class_weight=class_weights, callbacks=callbacks, verbose=1
        )

    # Phase 2: fine-tune top layers
    if ft_epochs > 0 and unfreeze_layers > 0:
        print(f"[Train] Fine-tune top {unfreeze_layers} layer(s) for {ft_epochs} epoch(s)")
        unfreeze_top_layers(base, unfreeze_layers)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(ft_lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
        model.fit(
            train_ds, validation_data=val_ds, epochs=ft_epochs,
            class_weight=class_weights, callbacks=callbacks, verbose=1
        )

    # Test
    print("[Eval] Test set")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f}")

    # Save model and class order
    model.save(out_model_path)
    print(f"Saved: {out_model_path}")
    with open("class_names_saved.txt", "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")
    print("Wrote: class_names_saved.txt")


# ---------------------------
# Inference
# ---------------------------
def predict_one(model_path: Path, image_path: Path, class_names_path: Path, img_size: Tuple[int, int]) -> None:
    """
    Single-image prediction. Do NOT divide by 255. Model graph contains preprocess_input.
    """
    model = tf.keras.models.load_model(model_path)
    class_names = read_class_names(class_names_path)
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    x = tf.keras.utils.img_to_array(img)               # float32 0..255
    x = tf.expand_dims(x, 0)                           # (1, H, W, 3)
    probs = model.predict(x, verbose=0)[0]             # softmax
    top = int(np.argmax(probs))
    print("\nProbabilities:")
    for i, (c, p) in enumerate(zip(class_names, probs)):
        print(f"  {i:02d} {c:<12} {p:.4f}")
    print(f"\nPrediction: {class_names[top]}  p={probs[top]:.4f}")


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MobileNetV2 transfer learning with smoke test and cached pipelines.")
    p.add_argument("--data-root", type=Path, default=Path("data_splits"))
    p.add_argument("--img-size", type=int, nargs=2, default=(160, 160), metavar=("H", "W"))
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--freeze-epochs", type=int, default=2)
    p.add_argument("--ft-epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ft-lr", type=float, default=1e-4)
    p.add_argument("--unfreeze-layers", type=int, default=20)
    p.add_argument("--model", type=Path, default=Path("out_final.keras"))
    p.add_argument("--predict", type=Path, default=None)
    p.add_argument("--class-names", type=Path, default=Path("data_splits/class_names.txt"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    setup_device()

    if args.predict is not None and args.model.exists():
        predict_one(args.model, args.predict, args.class_names, tuple(args.img_size))
        return

    train(
        data_root=args.data_root,
        img_size=tuple(args.img_size),
        batch=args.batch,
        freeze_epochs=args.freeze_epochs,
        ft_epochs=args.ft_epochs,
        lr=args.lr,
        ft_lr=args.ft_lr,
        unfreeze_layers=args.unfreeze_layers,
        out_model_path=args.model,
    )


if __name__ == "__main__":
    main()
