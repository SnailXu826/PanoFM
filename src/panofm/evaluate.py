"""Tooth-level evaluation with fixed thresholds and radiograph bootstrap CIs."""

import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .data import DISEASES, identity, read_jsonl, read_teeth
from .predict import read_thresholds


def aligned_predictions(manifest, predictions):
    table = {}
    for row in read_jsonl(predictions):
        key = identity(row)
        scores = row["probabilities"]
        if key in table or scores.keys() != DISEASES.keys():
            raise ValueError("Predictions need unique tooth keys and scores for all six diseases")
        table[key] = [scores[name] for name in DISEASES]
    records = read_teeth(manifest)
    if table.keys() != {record.key for record in records}:
        raise ValueError("Predictions and manifest must contain the same tooth keys")
    records = [record for record in records if record.labels is not None]
    if not records:
        raise ValueError("Evaluation requires ground-truth labels")
    targets = np.array([[name in record.labels for name in DISEASES] for record in records])
    probabilities = np.asarray([table[record.key] for record in records], dtype=float)
    if not np.isfinite(probabilities).all() or probabilities.min() < 0 or probabilities.max() > 1:
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    return records, targets, probabilities


def select_thresholds(args):
    _, targets, scores = aligned_predictions(args.manifest, args.predictions)
    thresholds = {}
    for index, disease in enumerate(DISEASES):
        if np.unique(targets[:, index]).size != 2:
            raise ValueError(f"Threshold fitting needs both positive and negative examples for {disease}")
        fpr, tpr, values = roc_curve(targets[:, index], scores[:, index])
        # An infinite threshold is an ROC sentinel rather than an observed cutoff.
        thresholds[disease] = float(values[1 + np.argmax((tpr - fpr)[1:])])
    write_json(args.output, thresholds)


def metrics(target, score, threshold):
    prediction = score >= threshold
    positive = int(target.sum())
    negative = len(target) - positive
    return {
        "auc": float(roc_auc_score(target, score)) if positive and negative else None,
        "accuracy": float((prediction == target).mean()),
        "sensitivity": float(prediction[target].mean()) if positive else None,
        "specificity": float((~prediction[~target]).mean()) if negative else None,
    }


def summarize(records, targets, scores, thresholds, bootstrap, seed):
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.image_id].append(index)
    groups = list(groups.values())
    generator = np.random.default_rng(seed)
    distribution = defaultdict(list)
    for _ in range(bootstrap):
        sample = np.concatenate([groups[index] for index in generator.integers(0, len(groups), len(groups))])
        for index, disease in enumerate(DISEASES):
            values = metrics(targets[sample, index], scores[sample, index], thresholds[disease])
            for name, value in values.items():
                if value is not None:
                    distribution[disease, name].append(value)
    results = {}
    for index, disease in enumerate(DISEASES):
        result = metrics(targets[:, index], scores[:, index], thresholds[disease])
        confidence = {}
        for name in result:
            values = distribution[disease, name]
            confidence[name] = np.percentile(values, [2.5, 97.5]).tolist() if len(values) >= 2 else None
        results[disease] = {"n_positive": int(targets[:, index].sum()), **result, "ci95": confidence}
    aucs = [result["auc"] for result in results.values() if result["auc"] is not None]
    return {"n_teeth": len(records), "n_radiographs": len(groups),
            "macro_auc": float(np.mean(aucs)) if aucs else None, "diseases": results}


def evaluate(args):
    if args.bootstrap < 0:
        raise ValueError("Use a nonnegative number of bootstrap replicates")
    records, targets, scores = aligned_predictions(args.manifest, args.predictions)
    thresholds = read_thresholds(args.thresholds)
    result = {"overall": summarize(records, targets, scores, thresholds, args.bootstrap, args.seed)}
    centers = sorted({record.center for record in records if record.center is not None})
    if centers:
        result["by_center"] = {}
        for center in centers:
            indices = [index for index, record in enumerate(records) if record.center == center]
            result["by_center"][center] = summarize([records[i] for i in indices], targets[indices],
                                                     scores[indices], thresholds, args.bootstrap, args.seed)
    write_json(args.output, result)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
