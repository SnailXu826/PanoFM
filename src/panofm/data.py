"""Tooth manifests, permanent tooth numbering, and image loading."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

DISEASES = {
    "PP": "periapical periodontitis",
    "DC": "dental caries",
    "DF": "dental filling",
    "IT": "impacted tooth",
    "TAC": "tooth after crown restoration",
    "TAR": "tooth after root canal treatment",
}
FDI_CODES = tuple(10 * quadrant + position for quadrant in range(1, 5) for position in range(1, 9))
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def identity(row: dict) -> tuple[str, int]:
    image_id, fdi = row["image_id"], row["fdi"]
    if not isinstance(image_id, str) or not image_id.strip():
        raise ValueError("image_id must be a nonempty pseudonymous string")
    if type(fdi) is not int or fdi not in FDI_CODES:
        raise ValueError("fdi must be a permanent tooth code from quadrant 1–4 and position 1–8")
    return image_id, fdi


@dataclass(frozen=True)
class Tooth:
    image_id: str
    fdi: int
    path: Path
    labels: tuple[str, ...] | None
    center: str | None

    @property
    def key(self) -> tuple[str, int]:
        return self.image_id, self.fdi


def read_teeth(manifest: Path) -> list[Tooth]:
    records = []
    seen = set()
    for row in read_jsonl(manifest):
        key = identity(row)
        if key in seen:
            raise ValueError("Duplicate image_id and fdi in tooth manifest")
        seen.add(key)
        labels = row.get("labels")
        if labels is not None and (
            not isinstance(labels, list)
            or any(not isinstance(label, str) or label not in DISEASES for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("labels must contain distinct disease codes")
        center = row.get("center")
        if center is not None and (not isinstance(center, str) or not center.strip()):
            raise ValueError("center must be a nonempty pseudonymous string when present")
        records.append(Tooth(*key, manifest.parent / row["path"],
                             tuple(labels) if labels is not None else None, center))
    if not records:
        raise ValueError("The tooth manifest is empty")
    return records


def read_context(path: Path) -> dict[tuple[str, int], str]:
    records = {}
    for row in read_jsonl(path):
        key = identity(row)
        if key in records or not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("Context entries must be unique and contain nonempty text")
        records[key] = row["text"]
    return records


def load_crop(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(np.array(gray, dtype=np.float32)).div_(255)
    return pixels.expand(3, -1, -1).sub(MEAN).div(STD)


class ToothDataset(Dataset):
    def __init__(self, records: list[Tooth], size: int, context: dict | None = None):
        if not records:
            raise ValueError("No teeth satisfy the requirements for this stage")
        if context is not None and any(record.key not in context for record in records):
            raise ValueError("The context cache must cover every tooth in this dataset")
        self.records = records
        self.size = size
        self.context = context
        self.arch_ids = {key: i for i, key in enumerate(dict.fromkeys(r.image_id for r in records))}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        labels = record.labels or ()
        return {
            "image": load_crop(record.path, self.size),
            "arch": self.arch_ids[record.image_id],
            "fdi": record.fdi,
            "index": index,
            "labels": torch.tensor([disease in labels for disease in DISEASES], dtype=torch.float32),
            "text": self.context[record.key] if self.context is not None else "",
        }
