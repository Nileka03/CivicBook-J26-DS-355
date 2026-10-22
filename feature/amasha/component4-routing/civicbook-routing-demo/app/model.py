"""
Loads the trained Component 4 model (the same architecture as the Kaggle notebook).

Files expected in the model folder (download them from the Kaggle version's Output tab):
    best_model.pt          trained weights
    routing_config.json    labels and thresholds
    tokenizer.json, tokenizer_config.json (and any other tokenizer files saved with them)
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, XLMRobertaConfig

log = logging.getLogger("component4")

DROPOUT = 0.1

# Used only if the model's config cannot be downloaded (no internet on first run).
XLMR_BASE_CONFIG = dict(vocab_size=250002, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072,
                        max_position_embeddings=514, type_vocab_size=1,
                        layer_norm_eps=1e-5, pad_token_id=1, bos_token_id=0, eos_token_id=2)


class RoutingModel(nn.Module):
    """XLM-R encoder + BIO head + institution head + one department head per institution.
    Must match the notebook exactly, so the saved weights load."""

    def __init__(self, encoder_config, institutions, departments, n_tags):
        super().__init__()
        self.encoder = AutoModel.from_config(encoder_config)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(DROPOUT)
        self.bio_head = nn.Linear(hidden, n_tags)
        self.inst_head = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(DROPOUT),
                                       nn.Linear(hidden, len(institutions)))
        self.dept_heads = nn.ModuleDict({
            inst: nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(DROPOUT),
                                nn.Linear(hidden, len(departments[inst])))
            for inst in institutions})


def _encoder_config(model_name):
    try:
        return AutoConfig.from_pretrained(model_name)
    except Exception as e:                                   # offline
        if "large" in str(model_name):
            raise RuntimeError("Could not load the xlm-roberta-large config; connect to the "
                               "internet once so it can be downloaded.") from e
        log.warning("Could not download the encoder config (%s); using built-in "
                    "xlm-roberta-base settings.", e)
        return XLMRobertaConfig(**XLMR_BASE_CONFIG)


def load(model_dir: Path):
    model_dir = Path(model_dir)
    cfg_path = model_dir / "routing_config.json"
    weights = model_dir / "best_model.pt"
    for p in (cfg_path, weights):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Copy the Kaggle model files into {model_dir}.")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = RoutingModel(_encoder_config(cfg["model_name"]), cfg["institutions"],
                         cfg["departments"], len(cfg["tags"]))
    state = torch.load(weights, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(f"Weights do not match the model. Missing: {missing[:5]} "
                           f"Unexpected: {list(unexpected)[:5]}")
    model.eval()
    log.info("Loaded %s from %s", cfg["model_name"], model_dir)
    return model, tokenizer, cfg
