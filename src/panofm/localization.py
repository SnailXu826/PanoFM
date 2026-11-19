"""Panoramic preprocessing, Mask R-CNN tooth localization, and crop export."""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from .data import FDI_CODES, identity, read_jsonl, write_jsonl
from .runtime import load_checkpoint, save_checkpoint

PANORAMA_SIZE = (1536, 768)


def read_panorama(path: Path) -> Image.Image:
    if path.suffix.lower() in (".dcm", ".dicom", ""):
        import pydicom
        from pydicom.pixels import apply_modality_lut

        dicom = pydicom.dcmread(path)
        data = np.asarray(apply_modality_lut(dicom.pixel_array, dicom), dtype=np.float32)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = -data
    else:
        with Image.open(path) as image:
            if image.mode in ("L", "I", "F") or image.mode.startswith("I;16"):
                data = np.array(image, dtype=np.float32)
            else:
                data = np.array(image.convert("L"), dtype=np.float32)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError("Expected a finite single-frame grayscale radiograph")
    low, high = np.percentile(data, (0.5, 99.5))
    data = np.clip((data - low) / (high - low), 0, 1) if high > low else np.zeros_like(data)
    return Image.fromarray(data.astype(np.float32)).resize(PANORAMA_SIZE, Image.Resampling.BILINEAR)


def crop_tooth(panorama: Image.Image, box, image_size: int) -> Image.Image:
    x1, y1, x2, y2 = clip_box(box, panorama.size)
    mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
    width, height = panorama.size
    box = (max(0, math.floor(x1 - mx)), max(0, math.floor(y1 - my)),
           min(width, math.ceil(x2 + mx)), min(height, math.ceil(y2 + my)))
    crop = panorama.crop(box)
    side = max(crop.size)
    square = Image.new("F", (side, side))
    square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    resized = np.clip(np.asarray(square.resize((image_size, image_size), Image.Resampling.BILINEAR)), 0, 1)
    return Image.fromarray(np.rint(resized * 255).astype(np.uint8))


def clip_box(box, size):
    x1, y1, x2, y2 = map(float, box)
    if not all(map(math.isfinite, (x1, y1, x2, y2))):
        raise ValueError("Tooth box coordinates must be finite")
    width, height = size
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if not (x1 < x2 and y1 < y2):
        raise ValueError("Tooth box must have positive area inside the radiograph")
    return x1, y1, x2, y2


def detector_input(image: Image.Image):
    return torch.from_numpy(np.array(image, dtype=np.float32)).expand(3, -1, -1).contiguous()


def detector():
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    return maskrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=len(FDI_CODES) + 1,
                                 min_size=PANORAMA_SIZE[1], max_size=PANORAMA_SIZE[0])


def read_panoramas(manifest):
    rows = read_jsonl(manifest)
    image_ids = [identity({"image_id": row["image_id"], "fdi": 11})[0] for row in rows]
    if not rows or len(set(image_ids)) != len(rows):
        raise ValueError("Use a unique image_id per radiograph in a nonempty manifest")
    return rows


class LocalizationDataset(Dataset):
    def __init__(self, manifest):
        self.rows = read_panoramas(manifest)
        self.root = manifest.parent

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        record = self.rows[index]
        image = read_panorama(self.root / record["path"])
        labels, masks, boxes = [], [], []
        seen = set()
        for tooth in record["teeth"]:
            _, fdi = identity({"image_id": record["image_id"], "fdi": tooth["fdi"]})
            if fdi in seen:
                raise ValueError("Duplicate FDI annotation in a radiograph")
            seen.add(fdi)
            mask = Image.new("L", image.size)
            polygon = [tuple(point) for point in tooth["polygon"]]
            if len(polygon) < 3 or any(not math.isfinite(v) for point in polygon for v in point):
                raise ValueError("A tooth polygon needs at least three finite coordinate pairs")
            ImageDraw.Draw(mask).polygon(polygon, fill=1)
            bounds = mask.getbbox()
            if bounds is None or bounds[2] - bounds[0] < 2 or bounds[3] - bounds[1] < 2:
                raise ValueError("A tooth polygon must enclose visible area")
            boxes.append(bounds)
            masks.append(torch.from_numpy(np.array(mask)))
            labels.append(FDI_CODES.index(fdi) + 1)
        targets = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "masks": torch.stack(masks) if masks else torch.empty((0, image.height, image.width), dtype=torch.uint8),
        }
        return detector_input(image), targets


def detection_collate(batch):
    return tuple(zip(*batch))


def train_localizer(args, cfg, rt):
    dataset = LocalizationDataset(args.manifest)
    model = detector().to(rt.device).train()
    batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=rt.workers, collate_fn=detection_collate)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    if cfg.training.epochs < 1:
        raise ValueError("At least one localization epoch is required")
    for epoch in range(cfg.training.epochs):
        running = torch.zeros((), device=rt.device)
        for images, targets in batches:
            optimizer.zero_grad(set_to_none=True)
            losses = model([image.to(rt.device) for image in images],
                           [{key: value.to(rt.device) for key, value in target.items()} for target in targets])
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            running += loss.detach()
        print(f"epoch={epoch + 1}/{cfg.training.epochs} loss={float(running) / len(batches):.6f}", flush=True)
    save_checkpoint(args.output, kind="localizer", model=model.state_dict())


@torch.inference_mode()
def locate(model, panorama, device, threshold):
    predictions = model([detector_input(panorama).to(device)])[0]
    best = {}
    for label, score, box, mask in zip(*(predictions[key].cpu() for key in ("labels", "scores", "boxes", "masks")), strict=True):
        fdi, score = FDI_CODES[int(label) - 1], float(score)
        if score < threshold or (fdi in best and score <= best[fdi][0]):
            continue
        ys, xs = torch.where(mask[0] >= 0.5)
        if len(xs):
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        else:
            box = box.tolist()
        best[fdi] = (score, box)
    return [{"fdi": fdi, "box": best[fdi][1]} for fdi in sorted(best)]


def crop_teeth(args, cfg, rt):
    if not 0 <= args.score_threshold <= 1:
        raise ValueError("Localization score threshold must be between 0 and 1")
    model = None
    if args.localizer:
        state = load_checkpoint(args.localizer, "localizer")
        model = detector()
        model.load_state_dict(state["model"])
        model.to(rt.device).eval()
    panoramas = read_panoramas(args.manifest)
    output = args.output
    if output.resolve() == args.manifest.resolve():
        raise ValueError("Use different input and output manifests")
    crop_dir = output.parent / (output.stem + "_crops")
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row in enumerate(panoramas):
        panorama = read_panorama(args.manifest.parent / row["path"])
        teeth = locate(model, panorama, rt.device, args.score_threshold) if model else row["teeth"]
        labels = {entry["fdi"]: entry["labels"] for entry in row.get("teeth", ()) if "labels" in entry}
        seen = set()
        for tooth in teeth:
            image_id, fdi = identity({"image_id": row["image_id"], "fdi": tooth["fdi"]})
            if fdi in seen:
                raise ValueError("Duplicate FDI for the same panoramic image")
            seen.add(fdi)
            filename = f"scan-{index:06d}-{fdi}.png"
            crop_tooth(panorama, tooth["box"], cfg.vision.image_size).save(crop_dir / filename)
            record = {"image_id": image_id, "fdi": fdi, "path": f"{crop_dir.name}/{filename}"}
            if fdi in labels:
                record["labels"] = labels[fdi]
            if "center" in row:
                record["center"] = row["center"]
            rows.append(record)
    if not rows:
        raise ValueError("No teeth were found in the supplied radiographs")
    write_jsonl(output, rows)
    print(f"Exported {len(rows)} tooth crops from {len(panoramas)} radiographs", flush=True)
