"""Training stages for the teacher, the two pretraining stages, and diagnosis."""

from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .data import DISEASES, ToothDataset, read_context, read_teeth
from .models.diagnosis import DiagnosisModel, bert_resources
from .models.prototype import DinoTeacher, PrototypePretrainer
from .models.vision import InstancePretrainer, VisionEncoder
from .runtime import ArchBatches, load_checkpoint, loader, loss_mean, save_checkpoint, set_epoch


def run_epochs(model, batches, rt, cfg, step, after_step=None):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay,
    )
    if cfg.epochs < 1:
        raise ValueError("At least one training epoch is required")
    model.train()
    for epoch in range(cfg.epochs):
        set_epoch(batches, epoch)
        total, count = torch.zeros((), device=rt.device), 0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            with rt.autocast():
                loss = step(model, batch)
            loss.backward()
            optimizer.step()
            if after_step is not None:
                after_step(batch)
            total += loss.detach()
            count += 1
        if rt.primary:
            print(f"epoch={epoch + 1}/{cfg.epochs} loss={float(total) / count:.6f}", flush=True)


def encoder_state(encoder, state, cfg):
    if state["vision"] != asdict(cfg.vision):
        raise ValueError("The checkpoint and configuration specify different vision architectures")
    encoder.load_state_dict(state["encoder"])


def pretrain_instance(args, cfg: Config, rt):
    dataset = ToothDataset(read_teeth(args.manifest), cfg.vision.image_size)
    sampler = ArchBatches(dataset.records, rt.local_batch(cfg.instance.batch_size), rt)
    batches = loader(dataset, rt, batch_sampler=sampler)
    core = InstancePretrainer(VisionEncoder(cfg.vision), cfg.instance)
    unused = not all((cfg.instance.contrastive_weight, cfg.instance.position_weight, cfg.instance.mim_weight))
    model = rt.model(core, unused=unused)
    core.update_momentum(0)

    def step(model, batch):
        losses = model(batch["image"].to(rt.device), batch["arch"].to(rt.device), batch["fdi"].to(rt.device))
        return sum(losses.values())

    run_epochs(model, batches, rt, cfg.training, step, lambda _: core.update_momentum(cfg.instance.ema_momentum))
    if rt.primary:
        save_checkpoint(args.output, kind="instance", vision=asdict(cfg.vision), encoder=core.encoder.state_dict())


def single_disease_dataset(manifest, image_size):
    records = [record for record in read_teeth(manifest) if record.labels and len(record.labels) == 1]
    observed = {record.labels[0] for record in records}
    if observed != DISEASES.keys():
        raise ValueError("Teacher and prototype training require singly labeled examples for all six diseases")
    return ToothDataset(records, image_size)


def train_teacher(args, cfg: Config, rt):
    dataset = single_disease_dataset(args.manifest, cfg.vision.image_size)
    batches = loader(dataset, rt, rt.local_batch(cfg.prototype.batch_size), training=True)
    core = DinoTeacher.from_local(args.dino)
    model = rt.model(core)

    def step(model, batch):
        logits = model(batch["image"].to(rt.device))
        targets = batch["labels"].to(rt.device).argmax(dim=1)
        return F.cross_entropy(logits.float(), targets)

    # Section 3.1.2 uses a categorical objective for single-disease guidance.
    run_epochs(model, batches, rt, cfg.training, step)
    if rt.primary:
        save_checkpoint(args.output, kind="teacher", dino=core.model.config.to_dict(), model=core.state_dict())


@torch.no_grad()
def initialize_centroids(teacher, batches, bank, rt):
    if rt.primary:
        sums = torch.zeros_like(bank.centroids, dtype=torch.float64)
        counts = torch.zeros(len(bank.centroids), device=rt.device, dtype=torch.float64)
        for batch in batches:
            labels = batch["labels"].to(rt.device).argmax(dim=1)
            features = teacher.features(batch["image"].to(rt.device)).double()
            sums.index_add_(0, labels, features)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float64))
        bank.centroids.copy_(sums / counts[:, None])
    if rt.world_size > 1:
        torch.distributed.broadcast(bank.centroids, src=0)


def pretrain_prototype(args, cfg: Config, rt):
    dataset = single_disease_dataset(args.manifest, cfg.vision.image_size)
    local_batch = rt.local_batch(cfg.prototype.batch_size)
    batches = loader(dataset, rt, local_batch, training=True)
    teacher_state = load_checkpoint(args.teacher, "teacher")
    teacher = DinoTeacher.from_config(teacher_state["dino"])
    teacher.load_state_dict(teacher_state["model"])
    teacher.to(rt.device).eval().requires_grad_(False)
    if teacher.model.config.hidden_size != cfg.vision.width:
        raise ValueError("The teacher and PanoFM embeddings must have the same width")
    core = PrototypePretrainer(VisionEncoder(cfg.vision), cfg.prototype)
    encoder_state(core.encoder, load_checkpoint(args.init, "instance"), cfg)
    model = rt.model(core)
    initialize_centroids(teacher, loader(dataset, rt, local_batch) if rt.primary else None, core.bank, rt)

    def step(model, batch):
        images = batch["image"].to(rt.device)
        labels = batch["labels"].to(rt.device).argmax(dim=1)
        with torch.no_grad():
            references = teacher.features(images)
        return model(images, labels, references)

    @torch.no_grad()
    def update_prototypes(batch):
        features = core.encoder(batch["image"].to(rt.device))[0]
        core.bank.update(features, batch["labels"].to(rt.device).argmax(dim=1))

    run_epochs(model, batches, rt, cfg.training, step, update_prototypes)
    if rt.primary:
        save_checkpoint(args.output, kind="prototype", vision=asdict(cfg.vision), encoder=core.encoder.state_dict(),
                        prototypes=core.bank.state_dict(), classifier=core.classifier.state_dict())


def focal_loss(logits, targets, phase: int, alpha: float, gamma: float):
    # Phase 1 uses healthy teeth as the negatives for every head.
    targets = targets.float()
    logits = logits.float()
    eligible = ((targets == 1) | (targets.sum(dim=1, keepdim=True) == 0)) if phase == 1 else torch.ones_like(targets)
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    # 1 - exp(-BCE) evaluates 1 - p_t stably, including confident predictions.
    modulation = (-torch.expm1(-cross_entropy)).pow(gamma)
    weights = torch.where(targets == 1, alpha, 1 - alpha)
    return loss_mean((weights * modulation * cross_entropy * eligible).sum(), eligible.sum())


def tokenize(tokenizer, text, cfg, device):
    return tokenizer(list(text), padding=True, truncation=True, max_length=cfg.diagnosis.text_length,
                     return_tensors="pt").to(device)


@torch.no_grad()
def fit_alignment(model, dataset, cfg, rt):
    if rt.primary:
        model.eval()
        generator = torch.Generator().manual_seed(rt.seed)
        indices = torch.randperm(len(dataset), generator=generator)[:cfg.ot.fit_samples].tolist()
        subset = torch.utils.data.Subset(dataset, indices)
        tokens = [model.encoder(batch["image"].to(rt.device))[1].cpu()
                  for batch in loader(subset, rt, min(16, rt.local_batch(cfg.diagnosis.batch_size)))]
        model.alignment.fit_clusters(torch.cat(tokens), rt.seed)
    if rt.world_size > 1:
        torch.distributed.broadcast(model.alignment.cluster_index, src=0)


def train_diagnosis(args, cfg: Config, rt, *, context_model=False):
    records = [record for record in read_teeth(args.manifest) if record.labels is not None]
    phase = 2 if context_model else args.phase
    mode = "vision" if context_model else args.mode
    kind = "context" if context_model else "diagnosis"
    if not 0 <= cfg.training.focal_alpha <= 1 or cfg.training.focal_gamma < 0:
        raise ValueError("Invalid focal loss parameters")
    if phase == 1 and not any(record.labels == () for record in records):
        raise ValueError("Phase 1 requires explicitly labeled healthy teeth")
    previous = load_checkpoint(args.init, "instance", "prototype", "diagnosis") if args.init else None
    continuing = previous is not None and previous["kind"] == "diagnosis"
    if context_model and (previous is None or previous["kind"] != "prototype"):
        raise ValueError("The preliminary vision classifier requires prototype-pretrained weights")
    if not context_model and phase == 2 and not continuing and not args.one_phase:
        raise ValueError("Phase 2 requires a phase-1 checkpoint or the explicit --one-phase ablation")
    if continuing and (context_model or phase != 2 or previous["phase"] != 1 or previous["mode"] != mode):
        raise ValueError("Resume diagnosis phase 2 from phase 1 in the same mode")
    if continuing and any(previous[name] != asdict(getattr(cfg, name)) for name in ("vision", "diagnosis", "ot")):
        raise ValueError("The diagnostic architecture must match the phase-1 checkpoint")

    bert, tokenizer, cache = None, None, None
    if mode == "multimodal":
        if args.context is None:
            raise ValueError("Multimodal diagnosis requires --context")
        bert, tokenizer = bert_resources(args.bert, previous["bert"] if continuing else None)
        cache = read_context(args.context)
    dataset = ToothDataset(records, cfg.vision.image_size, cache)
    batches = loader(dataset, rt, rt.local_batch(cfg.diagnosis.batch_size), training=True)
    core = DiagnosisModel(cfg, mode, bert)
    if continuing:
        core.load_state_dict(previous["model"])
    elif previous is not None:
        encoder_state(core.encoder, previous, cfg)
    model = rt.model(core)
    if core.alignment is not None and not continuing:
        fit_alignment(core, dataset, cfg, rt)

    def step(model, batch):
        text = tokenize(tokenizer, batch["text"], cfg, rt.device) if tokenizer is not None else None
        logits = model(batch["image"].to(rt.device), batch["fdi"].to(rt.device), text)
        return focal_loss(logits, batch["labels"].to(rt.device), phase, cfg.training.focal_alpha, cfg.training.focal_gamma)

    run_epochs(model, batches, rt, cfg.training, step)
    if rt.primary:
        save_checkpoint(args.output, kind=kind, mode=mode, phase=phase, vision=asdict(cfg.vision),
                        diagnosis=asdict(cfg.diagnosis), ot=asdict(cfg.ot), model=core.state_dict(),
                        bert=bert.config.to_dict() if bert is not None else None)
