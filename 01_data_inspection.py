"""Inspect 96x96 grayscale faces. Run with Python 3.12.

Examples:
  python 01_data_inspection.py --data-dir "D:/datasets/faces"
  python 01_data_inspection.py --search-root "D:/datasets" --no-show

Supported folders: root/0/*.png ... root/3/*.png, or
root/train/0/*.png and root/test/0/*.png (nested folders allowed).
Flat folders with names such as 0_example.jpg are also supported.
Class folder names may also be Korean labels or the English aliases below.
Images are inspected without resizing or changing the source files.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

LABELS = {0: "분노", 1: "슬픔", 2: "기쁨", 3: "중립"}
ENGLISH = {0: "Anger", 1: "Sadness", 2: "Happiness", 3: "Neutral"}
ALIASES = {
    "0": 0, "분노": 0, "anger": 0, "angry": 0,
    "1": 1, "슬픔": 1, "sadness": 1, "sad": 1,
    "2": 2, "기쁨": 2, "happiness": 2, "happy": 2, "joy": 2,
    "3": 3, "중립": 3, "neutral": 3,
}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def image_paths(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def classify(path: Path, root: Path):
    # Nearest recognized ancestor wins.
    for part in reversed(path.relative_to(root).parts[:-1]):
        if part.lower() in ALIASES:
            return ALIASES[part.lower()]
    # Also support a --data-dir that points directly to a class folder.
    root_label = ALIASES.get(root.name.lower())
    if root_label is not None:
        return root_label
    # The supplied archive uses label_filename.jpg in a flat dataset folder.
    prefix, separator, _ = path.stem.partition("_")
    return int(prefix) if separator and prefix in {"0", "1", "2", "3", "4"} else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, help="Dataset root (recommended)")
    parser.add_argument("--search-root", type=Path, default=Path.cwd(), help="Auto-search root; default: current directory")
    parser.add_argument("--output-dir", type=Path, default=Path("inspection_results"))
    parser.add_argument("--samples", type=int, default=5, help="Samples per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-scheme", choices=["requested", "provided-archive"], default="requested",
                        help="provided-archive maps original sad label 4 to requested label 1")
    parser.add_argument("--no-show", action="store_true", help="Save charts without opening windows")
    args = parser.parse_args()
    if not 1 <= args.samples <= 20:
        parser.error("--samples must be between 1 and 20")
    root = (args.data_dir or args.search_root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Directory does not exist: {root}")

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, UnidentifiedImageError

    paths = image_paths(root)
    if not paths:
        parser.error(f"No image files found under {root}. Specify --data-dir.")
    grouped = {label: [] for label in LABELS}
    rows = []
    unknown = 0
    sizes, modes = Counter(), Counter()
    for path in paths:
        label = classify(path, root)
        raw_label = label
        mapping = {0: 0, 2: 2, 3: 3, 4: 1} if args.label_scheme == "provided-archive" else {i: i for i in LABELS}
        label = mapping.get(raw_label)
        unknown += label is None
        row = {"path": str(path), "raw_label": raw_label if raw_label is not None else "", "label": label if label is not None else "",
               "class_name": LABELS.get(label, "unclassified"), "width": "", "height": "",
               "mode": "", "status": "", "error": ""}
        try:
            with Image.open(path) as img:
                img.load()
                row.update(width=img.width, height=img.height, mode=img.mode)
                sizes[img.size] += 1
                modes[img.mode] += 1
                problems = []
                if img.size != (96, 96):
                    problems.append("size_mismatch")
                if img.mode != "L":
                    problems.append("not_8bit_grayscale")
                row["status"] = ";".join(problems) or "ok"
                if label is not None:
                    grouped[label].append(path)
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            row.update(status="unreadable", error=str(exc))
        rows.append(row)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "image_audit.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"탐색 경로: {root}\n이미지 파일: {len(paths)} / 라벨 미분류: {unknown}")
    for label, name in LABELS.items():
        total = sum(row["label"] == label for row in rows)
        print(f"{label}: {name} — 파일 {total}, 읽기 가능 {len(grouped[label])}, 읽기 실패 {total-len(grouped[label])}")
    print(f"크기 분포: {dict(sizes)}\n색상 모드 분포: {dict(modes)}")
    print(f"읽기 실패: {sum(r['status'] == 'unreadable' for r in rows)}")
    print(f"96x96 크기 불일치: {sum('size_mismatch' in r['status'] for r in rows)}")
    print(f"8비트 흑백(L) 모드 불일치: {sum('not_8bit_grayscale' in r['status'] for r in rows)}")
    if unknown:
        print("라벨 미분류 이미지는 클래스 그래프에서 제외됩니다. 클래스 폴더 이름을 확인하세요.")

    labels = [f"{i}: {ENGLISH[i]}" for i in LABELS]
    totals = [sum(r["label"] == i for r in rows) for i in LABELS]
    valid = [len(grouped[i]) for i in LABELS]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(LABELS)
    ax.bar([i - .2 for i in x], totals, width=.4, label="Image files", color="#6987bb")
    bars = ax.bar([i + .2 for i in x], valid, width=.4, label="Readable images", color="#56a69a")
    ax.bar_label(bars, padding=3)
    ax.set(xticks=x, xticklabels=labels, ylabel="Count", title="Face dataset: class distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "class_counts.png", dpi=160)

    rng = random.Random(args.seed)
    sample_fig, axes = plt.subplots(4, args.samples, figsize=(args.samples * 2.2, 9), squeeze=False)
    for label in LABELS:
        selected = rng.sample(grouped[label], min(args.samples, len(grouped[label])))
        for col, ax in enumerate(axes[label]):
            ax.axis("off")
            if col < len(selected):
                with Image.open(selected[col]) as img:
                    # L conversion is for visualization only; original mode is audited above.
                    ax.imshow(img.convert("L"), cmap="gray", vmin=0, vmax=255)
                    ax.set_title(f"{label}: {ENGLISH[label]}\n{img.width}x{img.height}, {img.mode}", fontsize=9)
            else:
                ax.text(.5, .5, f"{label}: {ENGLISH[label]}\nNo sample", ha="center", va="center")
    sample_fig.suptitle("Face dataset: random samples (displayed in grayscale)")
    sample_fig.tight_layout(rect=(0, 0, 1, .96))
    sample_fig.savefig(output / "sample_images.png", dpi=160)
    print(f"검사 CSV 및 그래프 저장: {output}")
    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
