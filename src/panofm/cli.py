"""Command-line entry points for the complete PanoFM pipeline."""

import argparse
from pathlib import Path

from .config import load_config
from .evaluate import evaluate, select_thresholds
from .localization import crop_teeth, train_localizer
from .predict import make_context, predict
from .runtime import runtime
from .train import pretrain_instance, pretrain_prototype, train_diagnosis, train_teacher


def arguments(argv=None):
    parser = argparse.ArgumentParser(prog="panofm", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def add(name, description, *, model_command=True):
        command = commands.add_parser(name, help=description)
        command.add_argument("--manifest", type=Path, required=True, help="Input JSONL manifest")
        command.add_argument("--output", type=Path, required=True, help="Destination artifact")
        if model_command:
            command.add_argument("--device", default="auto", help="PyTorch device or auto")
            command.add_argument("--workers", type=int, default=0, help="Data loader workers per process")
        return command

    def configuration(command):
        command.add_argument("--config", type=Path, help="Experiment TOML file")

    command = add("train-localizer", "Train ResNet-50-FPN Mask R-CNN on tooth polygons")
    configuration(command)
    command.add_argument("--batch-size", type=int, default=1)
    command.add_argument("--lr", type=float, default=0.001)

    command = add("crop-teeth", "Normalize panoramas and export square tooth crops")
    configuration(command)
    command.add_argument("--localizer", type=Path, help="Mask R-CNN checkpoint")
    command.add_argument("--score-threshold", type=float, default=0.5)

    configuration(add("pretrain-instance", "Train the three panoramic instance objectives"))

    command = add("train-teacher", "Fine-tune the DINOv2 disease teacher")
    configuration(command)
    command.add_argument("--dino", type=Path, required=True, help="Local DINOv2 model directory")

    command = add("pretrain-prototype", "Refine the ViT using disease prototypes")
    configuration(command)
    command.add_argument("--init", type=Path, required=True, help="Instance checkpoint")
    command.add_argument("--teacher", type=Path, required=True, help="Fine-tuned DINOv2 checkpoint")

    command = add("train-context", "Fine-tune the preliminary vision classifier")
    configuration(command)
    command.add_argument("--init", type=Path, required=True, help="Prototype checkpoint")

    command = add("make-context", "Generate contextual text with a local Qwen checkpoint")
    command.add_argument("--checkpoint", type=Path, required=True, help="Preliminary vision checkpoint")
    command.add_argument("--thresholds", type=Path, help="JSON probability cutoffs for preliminary findings")
    command.add_argument("--qwen", type=Path, help="Local Qwen2.5-Instruct model directory")
    command.add_argument("--qwen-device", default="auto", help="Generation device or auto to shard across devices")
    command.add_argument("--raw-prompts", action="store_true", help="Cache prompts for the prompt-only ablation")
    command.add_argument("--batch-size", type=int, default=128, help="Preliminary vision inference batch size")
    command.add_argument("--generation-batch-size", type=int, default=1)
    command.add_argument("--max-new-tokens", type=int, default=128)

    command = add("train-diagnosis", "Train one progressive diagnosis phase")
    configuration(command)
    command.add_argument("--phase", type=int, choices=(1, 2), required=True)
    command.add_argument("--mode", choices=("vision", "ot", "multimodal"), default="multimodal")
    command.add_argument("--init", type=Path, help="Encoder initialization or phase-1 checkpoint")
    command.add_argument("--one-phase", action="store_true", help="Use phase 2 directly for the ablation")
    command.add_argument("--context", type=Path, help="Semantic context JSONL")
    command.add_argument("--bert", type=Path, help="Local BERT model/tokenizer directory")

    command = add("predict", "Export six tooth-level probabilities per crop")
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--context", type=Path)
    command.add_argument("--bert", type=Path, help="Local tokenizer directory used in training")
    command.add_argument("--batch-size", type=int, default=128, help="Batch size for vision-only checkpoints")

    command = add("select-thresholds", "Fit Youden cutoffs on validation predictions", model_command=False)
    command.add_argument("--predictions", type=Path, required=True)

    command = add("evaluate", "Compute AUC, accuracy, sensitivity, specificity, and confidence intervals", model_command=False)
    command.add_argument("--predictions", type=Path, required=True)
    command.add_argument("--thresholds", type=Path)
    command.add_argument("--bootstrap", type=int, default=1000)
    command.add_argument("--seed", type=int, default=42)
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = arguments(argv)
    command = args.command
    if args.output.resolve() == args.manifest.resolve():
        parser.error("The output and input manifest must be different")
    if getattr(args, "predictions", None) is not None and args.output.resolve() == args.predictions.resolve():
        parser.error("The output and predictions must be different")
    if command == "evaluate":
        evaluate(args)
        return
    if command == "select-thresholds":
        select_thresholds(args)
        return
    if getattr(args, "batch_size", 1) < 1 or args.workers < 0:
        parser.error("Use a positive batch size and a nonnegative worker count")
    cfg = load_config(getattr(args, "config", None))
    training = {
        "pretrain-instance": pretrain_instance,
        "train-teacher": train_teacher,
        "pretrain-prototype": pretrain_prototype,
        "train-context": lambda args, cfg, rt: train_diagnosis(args, cfg, rt, context_model=True),
        "train-diagnosis": train_diagnosis,
    }
    with runtime(args.device, args.workers, cfg.training.seed, distributed=command in training) as rt:
        if command in training:
            training[command](args, cfg, rt)
        elif command == "train-localizer":
            train_localizer(args, cfg, rt)
        elif command == "crop-teeth":
            crop_teeth(args, cfg, rt)
        elif command == "make-context":
            make_context(args, rt)
        else:
            predict(args, rt)
