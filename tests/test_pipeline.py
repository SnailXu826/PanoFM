"""Offline end-to-end checks using tiny model snapshots and synthetic teeth."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from panofm.cli import main
from panofm.data import DISEASES, read_jsonl, write_jsonl
from panofm.localization import read_panorama


def tiny_models(directory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        BertConfig,
        BertModel,
        BertTokenizerFast,
        Dinov2Config,
        Dinov2Model,
        PreTrainedTokenizerFast,
        Qwen2Config,
        Qwen2ForCausalLM,
    )

    dino = Dinov2Model(Dinov2Config(
        image_size=32, patch_size=8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
    ))
    dino.save_pretrained(directory / "dino")
    vocab = directory / "vocab.txt"
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "synthetic", "analysis"]
    vocab.write_text("\n".join(words) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab))
    bert = BertModel(BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32), add_pooling_layer=False)
    tokenizer.save_pretrained(directory / "bert")
    bert.save_pretrained(directory / "bert")
    words = ["[PAD]", "[UNK]", "[EOS]", "synthetic", "analysis", "user", "assistant"]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="user", eos_token="[EOS]",
                                       unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.chat_template = "{% for m in messages %}{{m['role']}} {{m['content']}} {% endfor %}assistant"
    qwen = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(words), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, intermediate_size=32, pad_token_id=0, bos_token_id=5,
        eos_token_id=2, tie_word_embeddings=False,
    ))
    with torch.no_grad():
        for name, value in qwen.named_parameters():
            value.fill_(1 if "norm" in name or "embed_tokens" in name else 0)
        qwen.lm_head.weight[4].fill_(1)
    qwen.save_pretrained(directory / "qwen")
    tokenizer.save_pretrained(directory / "qwen")


class PipelineTest(unittest.TestCase):
    def test_dicom_monochrome_inversion(self):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic.dcm"
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            dicom = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
            dicom.Rows, dicom.Columns = 2, 4
            dicom.SamplesPerPixel = 1
            dicom.PhotometricInterpretation = "MONOCHROME1"
            dicom.BitsAllocated = dicom.BitsStored = 16
            dicom.HighBit, dicom.PixelRepresentation = 15, 0
            dicom.PixelData = (np.arange(8, dtype="<u2") * 10).tobytes()
            dicom.save_as(path)
            panorama = np.asarray(read_panorama(path))
            self.assertEqual(panorama.shape, (768, 1536))
            self.assertGreater(float(panorama[0, 0]), float(panorama[-1, -1]))

    def test_cpu_pipeline(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temp:
                self.run_pipeline(Path(temp))
        finally:
            torch.set_num_threads(previous_threads)

    def run_pipeline(self, directory):
        def path(name):
            return str(directory / name)

        tiny_models(directory)
        config = directory / "test.toml"
        config.write_text(
            "[vision]\nimage_size = 32\npatch_size = 8\nwidth = 16\ndepth = 1\nheads = 2\n"
            "[training]\nepochs = 1\n"
            "[instance]\nbatch_size = 8\n"
            "[prototype]\nbatch_size = 8\n"
            "[diagnosis]\nbatch_size = 8\nfusion_width = 16\nfusion_heads = 2\n"
            "fusion_depth = 1\nqueries = 2\ntext_length = 64\n"
            "[ot]\nclusters = 4\nprojection_width = 8\nfit_samples = 8\n",
            encoding="utf-8",
        )
        boxes = [[400, 300, 500, 420], [515, 310, 615, 430], [970, 305, 1080, 425], [1090, 310, 1190, 440]]
        generator = np.random.default_rng(32)
        for split in ("train", "validation", "test"):
            panoramas = []
            for scan in range(2):
                pixels = generator.integers(0, 256, (96, 192), dtype=np.uint8)
                file = f"{split}-{scan}.png"
                Image.fromarray(pixels).save(directory / file)
                labels = list(DISEASES)[scan * 3:scan * 3 + 3]
                teeth = [{"fdi": fdi, "box": box, "labels": [labels[i]] if i < 3 else []}
                         for i, (fdi, box) in enumerate(zip((11, 12, 21, 22), boxes, strict=True))]
                panoramas.append({"image_id": f"synthetic-{split}-{scan}", "path": file,
                                  "center": "synthetic-center", "teeth": teeth})
            write_jsonl(directory / f"{split}-panoramas.jsonl", panoramas)
            main(["crop-teeth", "--manifest", path(f"{split}-panoramas.jsonl"), "--output",
                  path(f"{split}.jsonl"), "--config", str(config), "--device", "cpu"])

        def train(command, output, *arguments):
            main([command, "--manifest", path("train.jsonl"), "--output", path(output),
                  "--config", str(config), "--device", "cpu", *arguments])

        train("pretrain-instance", "instance.pt")
        train("train-teacher", "teacher.pt", "--dino", path("dino"))
        train("pretrain-prototype", "prototype.pt", "--init", path("instance.pt"), "--teacher", path("teacher.pt"))
        train("train-context", "preliminary.pt", "--init", path("prototype.pt"))
        for split in ("train", "validation", "test"):
            main(["make-context", "--manifest", path(f"{split}.jsonl"), "--output", path(f"{split}-context.jsonl"),
                  "--checkpoint", path("preliminary.pt"), "--qwen", path("qwen"), "--device", "cpu",
                  "--qwen-device", "cpu", "--max-new-tokens", "2", "--generation-batch-size", "2"])
            self.assertTrue(all(row["text"] == "analysis analysis"
                                for row in read_jsonl(directory / f"{split}-context.jsonl")))
        options = ["--bert", path("bert"), "--context", path("train-context.jsonl")]
        train("train-diagnosis", "phase1.pt", "--phase", "1", "--init", path("prototype.pt"), *options)
        train("train-diagnosis", "phase2.pt", "--phase", "2", "--init", path("phase1.pt"), *options)
        for split in ("validation", "test"):
            main(["predict", "--manifest", path(f"{split}.jsonl"), "--checkpoint", path("phase2.pt"),
                  "--bert", path("bert"), "--context", path(f"{split}-context.jsonl"),
                  "--output", path(f"{split}-predictions.jsonl"), "--device", "cpu"])
        main(["select-thresholds", "--manifest", path("validation.jsonl"), "--predictions",
              path("validation-predictions.jsonl"), "--output", path("thresholds.json")])
        main(["evaluate", "--manifest", path("test.jsonl"), "--predictions", path("test-predictions.jsonl"),
              "--thresholds", path("thresholds.json"), "--bootstrap", "4", "--output", path("metrics.json")])
        report = json.loads((directory / "metrics.json").read_text())
        self.assertEqual(report["overall"]["n_teeth"], 8)
        self.assertEqual(report["overall"]["n_radiographs"], 2)
        self.assertEqual(set(report["overall"]["diseases"]), set(DISEASES))
        self.assertIn("synthetic-center", report["by_center"])
