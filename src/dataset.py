"""
Data loading and split utilities.

CSV format (data/trash-data/csv/*.csv):
    file_name, main_category, sub_category

Images are expected at data/trash-data/image/{file_name}.
"""
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_GROUP_RE = re.compile(r"^(.*)_\d+$")


def normalize_label(text: str) -> str:
    """Strip whitespace and normalize ㎝/cm variants."""
    text = text.replace("\xa0", " ").strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("cm", "㎝")
    return text


def _extract_group_id(file_name: str) -> str:
    stem = Path(file_name).stem
    m = _GROUP_RE.match(stem)
    return m.group(1) if m else stem


def _is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.convert("RGB")
        return True
    except Exception:
        return False


@dataclass
class SampleRecord:
    image_path: str
    file_name: str
    main_category: str
    sub_category: str   # normalized full label, e.g. "소파_1인용"
    group_id: str
    # Optional GDINO metadata populated by 01_extract_crops.py.
    # Schema: {"detection_success": bool, "fallback": bool,
    #          "score": float|None, "box": [x0,y0,x1,y1]|None,
    #          "label_en": str|None, "image_size": [W,H]}
    # CLIP code may safely ignore this field.
    dino_meta: dict | None = None

    def to_dict(self) -> dict:
        d = {
            "image_path": self.image_path,
            "file_name": self.file_name,
            "main_category": self.main_category,
            "sub_category": self.sub_category,
            "group_id": self.group_id,
        }
        if self.dino_meta is not None:
            d["dino_meta"] = self.dino_meta
        return d

    @staticmethod
    def from_dict(d: dict) -> "SampleRecord":
        return SampleRecord(
            image_path=d["image_path"],
            file_name=d["file_name"],
            main_category=d["main_category"],
            sub_category=d["sub_category"],
            group_id=d["group_id"],
            dino_meta=d.get("dino_meta"),
        )


def load_records(
    csv_dir: Path,
    image_dir: Path,
    include_categories: list[str] | None = None,
    verify_images: bool = True,
    progress: bool = True,
) -> list[SampleRecord]:
    """Load all CSVs from csv_dir and match images from image_dir.

    Args:
        csv_dir: directory containing *.csv annotation files
        image_dir: flat directory containing all JPG images
        include_categories: list of main_category values to include (None = all)
        verify_images: if True, open each image with PIL to confirm it decodes
            (~ms per image; on slow disks this dominates runtime). Set False
            when the dataset is already known to be valid.
        progress: show a tqdm progress bar over CSV rows when tqdm is available.
    """
    include_set = set(include_categories) if include_categories else None
    records: list[SampleRecord] = []

    rows: list[dict] = []
    for csv_path in sorted(csv_dir.glob("*.csv")):
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            rows.extend(csv.DictReader(f))

    iterable = rows
    if progress:
        try:
            from tqdm import tqdm
            iterable = tqdm(rows, desc="load_records", unit="row")
        except ImportError:
            pass

    for row in iterable:
        main = normalize_label(row.get("main_category", ""))
        if not main:
            continue
        if include_set and main not in include_set:
            continue

        file_name = row.get("file_name", "").strip()
        if not file_name:
            continue

        img_path = image_dir / file_name
        if not img_path.exists():
            continue
        if verify_images and not _is_readable(img_path):
            continue

        sub = normalize_label(row.get("sub_category", ""))
        group_id = _extract_group_id(file_name)

        records.append(SampleRecord(
            image_path=str(img_path),
            file_name=file_name,
            main_category=main,
            sub_category=sub,
            group_id=group_id,
        ))

    return records


def sample_per_category(
    records: list[SampleRecord],
    max_per_category: int,
    seed: int = 42,
) -> list[SampleRecord]:
    """For each sub_category, keep at most max_per_category records sampled randomly.

    Sampling is done at the individual record level. Group integrity within the
    sampled set is preserved because grouped_stratified_split rebuilds groups
    from whatever records it receives.
    """
    by_label: dict[str, list[SampleRecord]] = defaultdict(list)
    for rec in records:
        by_label[rec.sub_category].append(rec)

    rng = random.Random(seed)
    result: list[SampleRecord] = []
    for label, recs in sorted(by_label.items()):
        original_count = len(recs)
        if original_count > max_per_category:
            recs = rng.sample(recs, max_per_category)
            print(f"  [sample] {label}: {len(recs)}/{original_count} records kept")
        result.extend(recs)
    return result


def grouped_stratified_split(
    records: list[SampleRecord],
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[SampleRecord]]:
    """Split by group_id so images of the same object stay in one split."""
    by_group: dict[str, list[SampleRecord]] = defaultdict(list)
    for rec in records:
        by_group[rec.group_id].append(rec)

    label_to_groups: dict[str, list[str]] = defaultdict(list)
    for gid, items in by_group.items():
        labels = {r.sub_category for r in items}
        if len(labels) == 1:
            label_to_groups[next(iter(labels))].append(gid)

    rng = random.Random(seed)
    split_groups: dict[str, set] = {"train": set(), "val": set(), "test": set()}

    for label, groups in label_to_groups.items():
        groups = list(groups)
        rng.shuffle(groups)
        n = len(groups)
        if n <= 2:
            train_n, val_n = n, 0
        else:
            train_n = max(1, int(round(n * train_ratio)))
            val_n = int(round(n * val_ratio))
            if train_n + val_n >= n:
                val_n = max(0, n - train_n - 1)

        test_n = n - train_n - val_n
        if test_n <= 0 and n >= 3:
            test_n = 1
            val_n = max(0, val_n - 1) if val_n > 0 else 0
            if test_n + train_n + val_n > n:
                train_n -= 1

        split_groups["train"].update(groups[:train_n])
        split_groups["val"].update(groups[train_n:train_n + val_n])
        split_groups["test"].update(groups[train_n + val_n:])

    result: dict[str, list[SampleRecord]] = {"train": [], "val": [], "test": []}
    for rec in records:
        for split_name, gids in split_groups.items():
            if rec.group_id in gids:
                result[split_name].append(rec)
                break

    return result


def save_splits(path: Path, splits: dict[str, list[SampleRecord]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: [r.to_dict() for r in v] for k, v in splits.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_splits(path: Path) -> dict[str, list[SampleRecord]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return {k: [SampleRecord.from_dict(d) for d in v] for k, v in payload.items()}


def distribution(records: list[SampleRecord]) -> dict[str, int]:
    return dict(Counter(r.sub_category for r in records))
