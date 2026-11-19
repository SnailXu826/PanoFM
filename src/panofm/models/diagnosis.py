"""Binary disease heads and a BERT-conditioned query transformer."""

from pathlib import Path

import torch
from torch import nn

from ..config import Config
from ..data import DISEASES
from .anatomy import AnatomicalAlignment
from .vision import VisionEncoder


class QueryBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.query_text = nn.TransformerDecoderLayer(
            width, heads, width * 4, batch_first=True, norm_first=True, dropout=0, activation="gelu",
        )
        self.visual_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, queries, text, padding, visual):
        queries = self.query_text(queries, text, memory_key_padding_mask=padding)
        return queries + self.visual_attention(self.norm(queries), visual, visual, need_weights=False)[0]


class DiagnosisModel(nn.Module):
    def __init__(self, config: Config, mode: str, bert: nn.Module | None = None):
        super().__init__()
        self.mode = mode
        self.encoder = VisionEncoder(config.vision)
        width = config.diagnosis.fusion_width
        self.alignment = None
        if mode == "vision":
            self.project_visual = nn.Linear(config.vision.width, width)
        else:
            patches = self.encoder.patch_count
            self.alignment = AnatomicalAlignment(config.vision.width, patches, config.ot)
            self.project_visual = nn.Sequential(nn.LayerNorm(patches), nn.Linear(patches, width))
        self.bert = bert
        if mode == "multimodal":
            if bert is None:
                raise ValueError("The multimodal mode requires BERT")
            self.project_text = nn.Linear(bert.config.hidden_size, width)
            self.queries = nn.Parameter(torch.empty(1, config.diagnosis.queries, width))
            nn.init.normal_(self.queries, std=0.02)
            self.fusion = nn.ModuleList(
                QueryBlock(width, config.diagnosis.fusion_heads) for _ in range(config.diagnosis.fusion_depth)
            )
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, len(DISEASES)))

    def forward(self, image: torch.Tensor, fdi: torch.Tensor, text=None) -> torch.Tensor:
        cls, patches = self.encoder(image)
        visual = self.project_visual(self.alignment(patches, fdi)) if self.alignment is not None else None
        if self.mode == "vision":
            features = self.project_visual(cls)
        elif self.mode == "ot":
            features = visual.mean(dim=1)
        else:
            semantic = self.project_text(self.bert(**text).last_hidden_state)
            queries = self.queries.to(semantic).expand(len(image), -1, -1)
            for block in self.fusion:
                queries = block(queries, semantic, text["attention_mask"] == 0, visual)
            features = queries.mean(dim=1)
        return self.head(features)


def bert_resources(path: Path | None, config: dict | None = None):
    from transformers import AutoTokenizer, BertConfig, BertModel

    if path is None or not path.is_dir():
        raise ValueError("Provide a local BERT tokenizer directory with --bert")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if config is not None:
        bert = BertModel(BertConfig.from_dict(config), add_pooling_layer=False)
    else:
        bert = BertModel.from_pretrained(path, local_files_only=True, add_pooling_layer=False)
    return bert, tokenizer
