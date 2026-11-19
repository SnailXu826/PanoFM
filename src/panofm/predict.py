"""Checkpoint-based inference and preliminary context generation."""

import json

import numpy as np
import torch

from .config import Config, DiagnosisConfig, OTConfig, VisionConfig
from .context import LocalReasoner, context_prompts
from .data import DISEASES, ToothDataset, read_context, read_teeth, write_jsonl
from .models.diagnosis import DiagnosisModel, bert_resources
from .runtime import ArchBatches, load_checkpoint, loader
from .train import tokenize


def read_thresholds(path):
    thresholds = dict.fromkeys(DISEASES, 0.5)
    if path is not None:
        thresholds = json.loads(path.read_text(encoding="utf-8"))
        if thresholds.keys() != DISEASES.keys() or any(
            not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in thresholds.values()
        ):
            raise ValueError("Thresholds must contain one probability in [0, 1] per disease")
    return thresholds


@torch.inference_mode()
def probabilities(records, checkpoint, context, bert_path, batch_size, rt):
    cfg = Config(vision=VisionConfig(**checkpoint["vision"]),
                 diagnosis=DiagnosisConfig(**checkpoint["diagnosis"]),
                 ot=OTConfig(**checkpoint["ot"]))
    mode = checkpoint["mode"]
    bert, tokenizer, cache = None, None, None
    if mode == "multimodal":
        if context is None:
            raise ValueError("Provide the matching semantic context with --context")
        bert, tokenizer = bert_resources(bert_path, checkpoint["bert"])
        cache = read_context(context)
    model = DiagnosisModel(cfg, mode, bert)
    model.load_state_dict(checkpoint["model"])
    model.to(rt.device).eval()
    dataset = ToothDataset(records, cfg.vision.image_size, cache)
    # A radiograph defines the test-time OT batch, independent of CLI batching.
    sampler = ArchBatches(records) if model.alignment is not None else None
    scores = np.empty((len(dataset), len(DISEASES)), dtype=np.float32)
    for batch in loader(dataset, rt, batch_size, batch_sampler=sampler):
        text = tokenize(tokenizer, batch["text"], cfg, rt.device) if tokenizer is not None else None
        logits = model(batch["image"].to(rt.device), batch["fdi"].to(rt.device), text)
        scores[batch["index"].numpy()] = logits.float().sigmoid().cpu().numpy()
    return scores


def make_context(args, rt):
    records = read_teeth(args.manifest)
    checkpoint = load_checkpoint(args.checkpoint, "context")
    scores = probabilities(records, checkpoint, None, None, args.batch_size, rt)
    prompts = context_prompts(records, scores, read_thresholds(args.thresholds))
    del checkpoint
    if args.raw_prompts:
        text = prompts
    else:
        if args.qwen is None:
            raise ValueError("Provide the local Qwen model with --qwen")
        if args.generation_batch_size < 1 or args.max_new_tokens < 1:
            raise ValueError("Generation batch size and maximum tokens must be positive")
        reasoner = LocalReasoner(args.qwen, args.qwen_device)
        text = []
        for start in range(0, len(prompts), args.generation_batch_size):
            text.extend(reasoner.generate(prompts[start:start + args.generation_batch_size], args.max_new_tokens))
    rows = ({"image_id": record.image_id, "fdi": record.fdi, "text": value}
            for record, value in zip(records, text, strict=True))
    write_jsonl(args.output, rows)


def predict(args, rt):
    records = read_teeth(args.manifest)
    checkpoint = load_checkpoint(args.checkpoint, "diagnosis", "context")
    scores = probabilities(records, checkpoint, args.context, args.bert, args.batch_size, rt)
    rows = ({"image_id": record.image_id, "fdi": record.fdi,
             "probabilities": dict(zip(DISEASES, values.tolist(), strict=True))}
            for record, values in zip(records, scores, strict=True))
    write_jsonl(args.output, rows)
