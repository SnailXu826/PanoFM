# PanoFM

PanoFM learns panoramic radiograph representations and predicts six findings for each permanent tooth. This repository implements the two pretraining stages, anatomical alignment, and offline clinical-context fusion described in *PanoFM: An LLM-empowered panoramic foundation model with clinical semantic integration for comprehensive dental disease diagnosis*. [Pattern Recognition, 180, 114411](https://doi.org/10.1016/j.patcog.2026.114411)

| Output | Finding |
| --- | --- |
| PP | Periapical periodontitis |
| DC | Dental caries |
| DF | Dental filling |
| IT | Impacted tooth |
| TAC | Tooth after crown restoration |
| TAR | Tooth after root canal treatment |

Each output is an independent probability. This is research software for tooth-level evaluation.

## Installation

Use Python 3.11 or later and a compatible PyTorch/Torchvision build for your device. Run the commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
panofm --help
```

Place the model files in local directories such as `weights/dinov2-base/`, `weights/bert-base/`, and `weights/qwen2.5-72b-instruct/`. Use local Transformers-format snapshots of [DINOv2 ViT-B/14](https://huggingface.co/facebook/dinov2-base), [BERT](https://huggingface.co/google-bert/bert-base-uncased), and [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct). The pipeline loads local paths only. For Qwen 72B, `--qwen-device auto` uses Accelerate to shard the model across available devices.

## Data

Use UTF-8 JSON Lines and place each split in its own manifest. Paths are relative to the manifest; absolute paths can be used in a local manifest. Use pseudonymous image and center IDs. Keep source radiographs, annotations, weights, predictions, and generated context in local directories such as `data/`, `runs/`, and `weights/`.

A panorama record contains an image path and optional tooth annotations. Use permanent FDI codes `11–18`, `21–28`, `31–38`, and `41–48`. Use `polygon` for localization training and `box` for manual crops. Provide coordinates on the resized 1536 × 768 image with the image's upper-left corner as the origin. Boxes are `[x_min, y_min, x_max, y_max]`. Examples throughout this README use synthetic IDs and coordinates.

```json
{"image_id":"example-001","path":"panoramas/example-001.dcm","center":"site-a","teeth":[{"fdi":26,"polygon":[[855,274],[912,274],[912,368],[855,368]],"box":[855,274,912,368],"labels":["DC"]}]}
```

A tooth manifest contains one row per present tooth. `labels: []` denotes an annotated healthy tooth. Omit `labels` for unlabeled teeth. Multiple findings are supported during diagnosis. DINOv2 guidance and prototype fitting use teeth with exactly one of the six findings and require examples for every class. The optional `center` field enables per-center metrics.

```json
{"image_id":"example-001","fdi":26,"path":"crops/scan-000000-26.png","labels":["DC"],"center":"site-a"}
{"image_id":"example-001","fdi":27,"path":"crops/scan-000000-27.png","labels":[],"center":"site-a"}
```

Use subject-disjoint training, validation, and test manifests. Preliminary context is generated from the vision model's predictions for other teeth on the same radiograph. The prompt includes only FDI codes and predicted finding names. Cached context is joined on `image_id` and `fdi`. Source DICOM metadata is used for pixel decoding and is excluded from exported crops. Compressed DICOM can require an additional pixel decoder.

## Train

### 1. Localize and crop teeth

```bash
panofm train-localizer --manifest data/localization-train.jsonl --output runs/localizer.pt
panofm crop-teeth --manifest data/panoramas-train.jsonl --localizer runs/localizer.pt \
  --output data/train.jsonl
```

Repeat cropping for the unlabeled, validation, and test sets. Omit `--localizer` when the manifest already contains FDI-tagged boxes. Existing 224 × 224 grayscale tooth crops can be used directly with a tooth manifest. The crop pipeline clips intensities at the 0.5th and 99.5th percentiles, adds a 10% margin, pads to a square, and resizes the crop. The loader replicates grayscale into three channels and applies ImageNet normalization.

### 2. Pretrain the panoramic encoder

```bash
panofm pretrain-instance --config configs/paper.toml --manifest data/unlabeled.jsonl \
  --output runs/instance.pt
panofm train-teacher --config configs/paper.toml --manifest data/train.jsonl \
  --dino weights/dinov2-base --output runs/teacher.pt
panofm pretrain-prototype --config configs/paper.toml --manifest data/train.jsonl \
  --init runs/instance.pt --teacher runs/teacher.pt --output runs/prototype.pt
```

The first stage combines multi-positive InfoNCE, relative mesial-distal order prediction, and masked pixel reconstruction. The second stage fits six disease centroids from a fine-tuned DINOv2 teacher and updates them with the Lyapunov acceptance rule. The resulting encoder can initialize other tooth-level heads.

### 3. Generate local semantic context

```bash
panofm train-context --config configs/paper.toml --manifest data/train.jsonl \
  --init runs/prototype.pt --output runs/preliminary.pt

for split in train validation test; do
  panofm make-context --manifest "data/${split}.jsonl" --checkpoint runs/preliminary.pt \
    --qwen weights/qwen2.5-72b-instruct --output "runs/${split}-context.jsonl"
done
```

The LLM receives text only and always sees the target finding as unknown. A preliminary classifier with calibrated validation cutoffs can pass them via `--thresholds`. Use `--raw-prompts` to reproduce the input format for the prompt-only ablation.

### 4. Train the two diagnosis phases

```bash
panofm train-diagnosis --config configs/paper.toml --manifest data/train.jsonl \
  --phase 1 --init runs/prototype.pt --bert weights/bert-base \
  --context runs/train-context.jsonl --output runs/diagnosis-phase1.pt

panofm train-diagnosis --config configs/paper.toml --manifest data/train.jsonl \
  --phase 2 --init runs/diagnosis-phase1.pt --bert weights/bert-base \
  --context runs/train-context.jsonl --output runs/diagnosis.pt
```

Phase 1 uses diseased positives and healthy negatives for each head. Phase 2 continues from phase 1 and includes teeth with other findings as negatives for that head. Both phases use focal loss. The visual-only and visual-plus-OT ablations use `--mode vision` and `--mode ot`; these modes can omit the context and BERT arguments. Use `--phase 2 --one-phase` to train a single phase directly. To ablate an instance objective, set its weight to zero in the TOML configuration.

The five representation and diagnosis training commands accept `torchrun`. Batch sizes are global. Instance pretraining packs complete radiographs within each rank, so the per-rank batch must hold the largest radiograph.

```bash
torchrun --standalone --nproc-per-node=8 -m panofm pretrain-instance \
  --config configs/paper.toml --manifest data/unlabeled.jsonl --output runs/instance.pt
```

## Predict and evaluate

Generate probabilities on a held-out validation split and fit Youden cutoffs there. Apply the same cutoffs to each test center.

```bash
for split in validation test; do
  panofm predict --manifest "data/${split}.jsonl" --checkpoint runs/diagnosis.pt \
    --context "runs/${split}-context.jsonl" --bert weights/bert-base \
    --output "runs/${split}-predictions.jsonl"
done

panofm select-thresholds --manifest data/validation.jsonl \
  --predictions runs/validation-predictions.jsonl --output runs/thresholds.json

panofm evaluate --manifest data/test.jsonl --predictions runs/test-predictions.jsonl \
  --thresholds runs/thresholds.json --output runs/test-metrics.json
```

Metrics include ROC AUC, accuracy, sensitivity, specificity, and 95% confidence intervals from bootstrap resampling by radiograph. Metrics without class support are represented as `null`. Evaluation uses a probability cutoff of 0.5 when a threshold file is omitted.

## Implementation notes

- `configs/paper.toml` carries the manuscript's reported image size, ViT layout, batch sizes, mask ratio, Adam and focal loss settings, and Sinkhorn settings. The contrastive temperature, momentum coefficient, distillation weights, initial prototype step size, and QFormer dimensions are configurable choices for details the manuscript leaves open.
- Equations 14–20 describe 64 projected channels and a transport plan indexed by unprojected anatomy channels and visual clusters. A balanced 64 × 64 plan with uniform marginals gives each cell a mass of at most 1/64, below the reported 0.05 threshold. This implementation uses 64 coordinates in the shared alignment space and computes a 5 × 16 plan over the FDI attributes and visual clusters. It thresholds the raw transport mass at 0.05 and uses the strongest match for any empty anatomy row. Training uses a binary mask in the forward pass and soft-plan gradients for the alignment projections. Inference computes the mean cost over all present teeth in one radiograph.
- Mask R-CNN predicts 32 permanent FDI instance classes plus background. The localization training recipe and the precise implementation of the query-based fusion adapter are release choices for details not specified by the paper.
- The clinical cohort, original splits, checkpoints and published comparison models are external research assets. The code supports the PanoFM stages and its principal ablations. Published cohort metrics require the corresponding data and model weights.

## Verification

```bash
python -m unittest discover -s tests -v
```

The integration test creates synthetic radiographs and tiny local DINOv2, BERT, and Qwen models in a temporary directory. It exercises cropping, training, offline context, prediction, and evaluation without downloading weights.
