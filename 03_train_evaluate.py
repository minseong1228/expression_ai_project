"""Train/evaluate a four-class pretrained ResNet18 using split CSVs.

Run: python 03_train_evaluate.py --epochs 30 --no-show
Evaluate: python 03_train_evaluate.py --evaluate-only --no-show
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

CLASS_NAMES = ["분노", "슬픔", "기쁨", "중립"]
PLOT_NAMES = ["Anger (0)", "Sadness (1)", "Happiness (2)", "Neutral (3)"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def read_split(csv_path: Path):
    rows = []
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires path,label columns: {csv_path}")
        for line, row in enumerate(reader, 2):
            label = int(row["label"])
            path = Path(row["path"]).expanduser()
            if not path.is_absolute():
                path = csv_path.parent / path
            path = path.resolve()
            if label not in range(4) or not path.is_file():
                raise ValueError(f"Invalid label or missing image: {csv_path}:{line}: {row}")
            rows.append((str(path), label))
    if not rows or {label for _, label in rows} != set(range(4)):
        raise ValueError(f"CSV must contain all four classes: {csv_path}")
    if len({path for path, _ in rows}) != len(rows):
        raise ValueError(f"Duplicate image paths: {csv_path}")
    return rows


def check_disjoint(splits):
    seen = set()
    for name, rows in splits.items():
        paths = {path for path, _ in rows}
        if seen & paths:
            raise ValueError(f"Image paths overlap across splits: {name}")
        seen.update(paths)


class FaceDataset:
    """Map-style dataset; defined at module level for Windows worker processes."""
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image
        path, label = self.rows[index]
        with Image.open(path) as image:
            # Replicate luminance into RGB so the pretrained conv1 remains intact.
            image = image.convert("L").convert("RGB")
            tensor = self.transform(image)
        return tensor, label, path


def seed_worker(worker_id):
    import torch
    import numpy as np
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_model(pretrained=True, freeze_backbone=False):
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 4)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None, freeze_backbone=False, collect=False):
    import torch
    training = optimizer is not None
    model.train(training)
    if training and freeze_backbone:
        # Keep frozen backbone BatchNorm running statistics fixed as well.
        model.eval()
        model.fc.train()
    loss_sum, correct, total = 0.0, 0, 0
    truths, predictions, paths = [], [], []
    with torch.set_grad_enabled(training):
        for images, labels, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered; training stopped.")
            if training:
                loss.backward()
                optimizer.step()
            predicted = logits.argmax(dim=1)
            count = labels.size(0)
            loss_sum += loss.item() * count
            correct += (predicted == labels).sum().item()
            total += count
            if collect:
                truths.extend(labels.cpu().tolist())
                predictions.extend(predicted.cpu().tolist())
                paths.extend(batch_paths)
    if total == 0:
        raise ValueError("Empty data loader")
    return loss_sum / total, correct / total, truths, predictions, paths


def overfit_streak(history, previous_streak, min_gap):
    if len(history) < 2:
        return 0
    previous, current = history[-2:]
    suspected = (current["train_loss"] < previous["train_loss"]
                 and current["val_loss"] > previous["val_loss"]
                 and current["train_acc"] - current["val_acc"] >= min_gap)
    return previous_streak + 1 if suspected else 0


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_history(output, history):
    with (output / "history.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def plot_results(output, history, truths, predictions, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    if history:
        epochs = [row["epoch"] for row in history]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for name in ("train", "val"):
            axes[0].plot(epochs, [r[f"{name}_loss"] for r in history], label=name)
            axes[1].plot(epochs, [r[f"{name}_acc"] for r in history], label=name)
        axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
        axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
        for ax in axes:
            ax.legend()
            ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(output / "learning_curves.png", dpi=180)
    matrix = confusion_matrix(truths, predictions, labels=list(range(4)))
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=PLOT_NAMES).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Test confusion matrix (best validation-loss model)")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true / predicted", *CLASS_NAMES])
        for name, counts in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([name, *counts])
    if show:
        plt.show()
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--csv-dir", type=Path, default=here / "dataset")
    parser.add_argument("--output-dir", type=Path, default=here / "training_results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only FC; default fine-tunes all layers")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping; 0 disables")
    parser.add_argument("--overfit-patience", type=int, default=3)
    parser.add_argument("--overfit-gap", type=float, default=0.05, help="Train/val accuracy gap (0..1)")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace previous training outputs")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.patience < 0
            or args.overfit_patience < 1 or not 0 <= args.overfit_gap <= 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        parser.error("Invalid training parameters")
    output = args.output_dir.expanduser().resolve()
    best_path = output / "best_model.pth"
    if args.evaluate_only and not best_path.is_file():
        parser.error(f"Best model not found: {best_path}")
    if not args.evaluate_only and output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error("Output directory is not empty. Use another --output-dir or --overwrite.")
    split_names = ["test"] if args.evaluate_only else ["train", "val", "test"]
    csv_dir = args.csv_dir.expanduser().resolve()
    splits = {name: read_split(csv_dir / f"{name}.csv") for name in split_names}
    check_disjoint(splits)

    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from sklearn.metrics import classification_report

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    # Original 96x96 faces -> RGB replication -> 224x224 -> ImageNet normalization.
    base = [transforms.Resize((224, 224), antialias=True)]
    tail = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    train_transform = transforms.Compose(base + [transforms.RandomHorizontalFlip(p=0.5)] + tail)
    eval_transform = transforms.Compose(base + tail)
    loaders = {}
    for name, rows in splits.items():
        generator = torch.Generator().manual_seed(args.seed)
        loaders[name] = DataLoader(FaceDataset(rows, train_transform if name == "train" else eval_transform),
                                   batch_size=args.batch_size, shuffle=name == "train",
                                   num_workers=args.workers, pin_memory=device.type == "cuda",
                                   worker_init_fn=seed_worker, generator=generator, drop_last=False)
        print(f"{name}: {len(rows):,} images", flush=True)
    model = build_model(pretrained=not args.evaluate_only, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    output.mkdir(parents=True, exist_ok=True)
    history = []
    print(f"Device: {device}", flush=True)
    if not args.evaluate_only:
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(class_names=CLASS_NAMES, input_size=[3, 224, 224], mean=MEAN, std=STD,
                      pretrained_weights="ResNet18_Weights.DEFAULT", torch_version=str(torch.__version__))
        save_json(output / "run_config.json", config)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=args.lr, weight_decay=args.weight_decay)
        best_loss, stale, streak = float("inf"), 0, 0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, *_ = run_epoch(model, loaders["train"], criterion, device, optimizer, args.freeze_backbone)
            val_loss, val_acc, *_ = run_epoch(model, loaders["val"], criterion, device)
            improved = val_loss < best_loss
            if improved:
                best_loss, stale = val_loss, 0
                # State dict only: portable, loadable with weights_only=True.
                temp = output / "best_model.pth.tmp"
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, temp)
                os.replace(temp, best_path)
                save_json(output / "best_model_metadata.json", {"epoch": epoch, "val_loss": val_loss,
                          "val_acc": val_acc, "class_names": CLASS_NAMES, "mean": MEAN, "std": STD,
                          "input_size": [3, 224, 224], "architecture": "resnet18"})
            else:
                stale += 1
            row = dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                       val_loss=val_loss, val_acc=val_acc, accuracy_gap=train_acc-val_acc)
            history.append(row)
            streak = overfit_streak(history, streak, args.overfit_gap)
            row["overfit_warning"] = streak >= args.overfit_patience
            save_history(output, history)
            print(f"Epoch {epoch:03d}/{args.epochs}: train loss={train_loss:.4f} acc={train_acc:.2%} | "
                  f"val loss={val_loss:.4f} acc={val_acc:.2%}" + (" [best saved]" if improved else ""), flush=True)
            if row["overfit_warning"]:
                print("Overfitting suspected: train loss decreases while val loss rises, with an accuracy gap.", flush=True)
            if args.patience and stale >= args.patience:
                print(f"Early stopping: validation loss did not improve for {stale} epochs.", flush=True)
                break
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_loss, test_acc, truths, predictions, paths = run_epoch(model, loaders["test"], criterion, device, collect=True)
    report = classification_report(truths, predictions, labels=list(range(4)), target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    save_json(output / "test_metrics.json", {"loss": test_loss, "accuracy": test_acc,
              "count": len(truths), "classification_report": report})
    with (output / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "true_label", "predicted_label"])
        writer.writerows(zip(paths, truths, predictions))
    if args.evaluate_only and (output / "history.csv").exists():
        with (output / "history.csv").open(encoding="utf-8-sig", newline="") as file:
            history = [{k: float(r[k]) for k in ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]}
                       for r in csv.DictReader(file)]
    print(f"Test (best model): loss={test_loss:.4f}, accuracy={test_acc:.2%}", flush=True)
    plot_results(output, history, truths, predictions, show=not args.no_show)
    print(f"Saved outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
