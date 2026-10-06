"""Python 3.12: stratified 70/15/15 split, without moving source images.

Stratification is performed independently within each class. Largest-remainder
rounding assigns whole images while keeping each class close to 70/15/15.
Only Python standard-library modules are required for this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

LABELS = {0: "분노", 1: "슬픔", 2: "기쁨", 3: "중립"}
ALIASES = {"0": 0, "분노": 0, "anger": 0, "angry": 0,
           "1": 1, "슬픔": 1, "sadness": 1, "sad": 1,
           "2": 2, "기쁨": 2, "happiness": 2, "happy": 2, "joy": 2,
           "3": 3, "중립": 3, "neutral": 3, "4": 4}
EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
RATIOS = (0.70, 0.15, 0.15)
SETS = ("train", "val", "test")


def raw_label(path: Path, root: Path):
    for part in reversed(path.relative_to(root).parts[:-1]):
        if part.lower() in ALIASES:
            return ALIASES[part.lower()]
    if root.name.lower() in ALIASES:
        return ALIASES[root.name.lower()]
    prefix, separator, _ = path.stem.partition("_")
    return int(prefix) if separator and prefix in {"0", "1", "2", "3", "4"} else None


def allocate(size: int) -> list[int]:
    exact = [size * ratio for ratio in RATIOS]
    counts = [math.floor(value) for value in exact]
    order = sorted(range(3), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[:size - sum(counts)]:
        counts[i] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--label-scheme", choices=["requested", "provided-archive"], default="requested")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing split CSVs and summary")
    args = parser.parse_args()
    root = args.data_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"데이터 폴더가 없습니다: {root}")
    mapping = {0: 0, 2: 2, 3: 3, 4: 1} if args.label_scheme == "provided-archive" else {i: i for i in LABELS}
    groups = {label: [] for label in LABELS}
    unknown = []
    seen = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        original = raw_label(path, root)
        label = mapping.get(original)
        if label is None:
            unknown.append(str(path))
            continue
        resolved = str(path.resolve())
        if resolved in seen:
            parser.error(f"동일한 실제 경로가 중복됩니다: {resolved}")
        seen.add(resolved)
        groups[label].append({"path": resolved, "label": label,
                              "class_name": LABELS[label], "raw_label": original})
    if unknown:
        parser.error(f"라벨을 판별할 수 없는 이미지 {len(unknown)}개. 예: {unknown[:3]}. "
                     "첨부 데이터는 --label-scheme provided-archive를 사용하세요.")
    for label, rows in groups.items():
        if not rows or min(allocate(len(rows))) == 0:
            parser.error(f"클래스 {label} ({LABELS[label]})의 이미지가 부족합니다: {len(rows)}개. "
                         "각 세트에 최소 1장이 배정되도록 데이터를 추가하세요.")
    targets = [output / f"{name}.csv" for name in SETS] + [output / "split_summary.json"]
    if not args.overwrite and any(path.exists() for path in targets):
        parser.error("결과가 이미 있습니다. 다른 --output-dir 또는 --overwrite를 사용하세요.")

    rng = random.Random(args.seed)
    splits = {name: [] for name in SETS}
    for label in LABELS:
        rows = groups[label].copy()
        rng.shuffle(rows)
        counts = allocate(len(rows))
        start = 0
        for name, count in zip(SETS, counts):
            splits[name].extend(rows[start:start + count])
            start += count
    for rows in splits.values():
        rng.shuffle(rows)
    # Fail before writing if the split ever loses or duplicates an input path.
    all_paths = [row["path"] for rows in splits.values() for row in rows]
    if len(all_paths) != len(seen) or set(all_paths) != seen:
        raise RuntimeError("분할 결과의 경로 중복 또는 누락이 발견되었습니다.")

    output.mkdir(parents=True, exist_ok=True)
    total = len(seen)
    summary = {"data_dir": str(root), "seed": args.seed, "label_scheme": args.label_scheme,
               "requested_ratios": dict(zip(SETS, RATIOS)), "total": total,
               "method": "per-class stratification with largest-remainder rounding", "sets": {}}
    print(f"전체: {total:,}개 / seed={args.seed}")
    print("세트       전체    분노(0) 슬픔(1) 기쁨(2) 중립(3)")
    for name, rows in splits.items():
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=["path", "label", "class_name", "raw_label"])
            writer.writeheader()
            writer.writerows(rows)
        counts = Counter(row["label"] for row in rows)
        summary["sets"][name] = {"count": len(rows), "ratio": len(rows) / total,
                                  "class_counts": {str(i): counts[i] for i in LABELS}}
        print(f"{name:<8} {len(rows):>6} " + " ".join(f"{counts[i]:>7}" for i in LABELS)
              + f"  ({len(rows)/total:.2%})")
    (output / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {output}")
    print("이미지 수의 정수 반올림으로 비율에 미세한 차이가 있을 수 있습니다.")


if __name__ == "__main__":
    main()
