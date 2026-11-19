"""Target-masked prompts and local-only Qwen generation."""

from collections import defaultdict
from pathlib import Path

import torch

from .data import DISEASES


def prompt_for_tooth(target: int, findings: dict[int, tuple[str, ...]]) -> str:
    lines = [f"The target is tooth {target} (FDI). Its condition is unknown."]
    for fdi, labels in sorted(findings.items()):
        if fdi != target:
            status = ", ".join(DISEASES[label] for label in labels) or "no finding predicted"
            lines.append(f"Tooth {fdi}: {status}.")
    if len(lines) == 1:
        lines.append("No other teeth were available for context.")
    lines.append("Based on the dental arch anatomy and co-occurrence of findings, describe relevant risks for the target tooth.")
    return "\n".join(lines)


def context_prompts(records, probabilities, thresholds) -> list[str]:
    by_image = defaultdict(dict)
    codes = tuple(DISEASES)
    for record, scores in zip(records, probabilities, strict=True):
        by_image[record.image_id][record.fdi] = tuple(
            code for code, probability in zip(codes, scores, strict=True) if probability >= thresholds[code]
        )
    return [prompt_for_tooth(record.fdi, by_image[record.image_id]) for record in records]


class LocalReasoner:
    def __init__(self, path: Path, device: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not path.is_dir():
            raise ValueError("Provide a local Qwen2.5-Instruct model directory")
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        options = {"local_files_only": True, "torch_dtype": "auto", "trust_remote_code": False}
        if device == "auto":
            options["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(path, **options).eval()
        if device != "auto":
            self.model.to(device)
        self.input_device = self.model.get_input_embeddings().weight.device

    @torch.inference_mode()
    def generate(self, prompts: list[str], max_tokens: int) -> list[str]:
        chats = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        encoded = self.tokenizer(
            chats, padding=True, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt",
        ).to(self.input_device)
        generated = self.model.generate(
            **encoded, max_new_tokens=max_tokens, do_sample=False, pad_token_id=self.tokenizer.pad_token_id,
        )
        text = self.tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        if any(not answer.strip() for answer in text):
            raise RuntimeError("Qwen returned an empty analysis. Check the local checkpoint and chat template.")
        return [answer.strip() for answer in text]
