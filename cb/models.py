"""Checkpoint download and loading.

Checkpoints (decided in DECISIONS.md, section "Checkpoints"):
  coconut      UdeSA/coconut-0.2B-gsm8k-aug, subfolder coconut/  (official recipe; card: 34.57%)
  cot          UdeSA/coconut-0.2B-gsm8k-aug, subfolder cot/      (its stage-0 CoT model; card: 43.14%)
  coconut_you  ModalityDance/latent-tts-coconut                  (You et al.'s checkpoint; optional regime check)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .gpt2 import FastGPT2

CHECKPOINTS = {
    "coconut": {"repo": "UdeSA/coconut-0.2B-gsm8k-aug", "sub": "coconut", "latent": True},
    "cot": {"repo": "UdeSA/coconut-0.2B-gsm8k-aug", "sub": "cot", "latent": False},
    "coconut_you": {"repo": "ModalityDance/latent-tts-coconut", "sub": "", "latent": True},
}
LATENT_TOKENS = ("<|latent|>", "<|start-latent|>", "<|end-latent|>")


@dataclass
class Bundle:
    name: str
    model: FastGPT2
    tok: object
    latent: bool
    eos_id: int
    start_id: int | None = None
    end_id: int | None = None
    latent_id: int | None = None
    path: Path | None = None


def download(name: str, cache_dir: str | Path) -> Path:
    from huggingface_hub import snapshot_download
    spec = CHECKPOINTS[name]
    pattern = f"{spec['sub']}/*" if spec["sub"] else None
    root = snapshot_download(spec["repo"], allow_patterns=pattern,
                             local_dir=str(Path(cache_dir) / spec["repo"].replace("/", "__")))
    return Path(root) / spec["sub"] if spec["sub"] else Path(root)


def read_state_dict(path: Path) -> dict:
    for fname in ("model.safetensors", "pytorch_model.bin", "model.pt", "checkpoint.pt"):
        f = path / fname
        if f.exists():
            if f.suffix == ".safetensors":
                from safetensors.torch import load_file
                return load_file(str(f))
            sd = torch.load(f, map_location="cpu", weights_only=True)
            return sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    raise FileNotFoundError(f"No weights in {path}")


def load(name: str, cache_dir: str | Path, device="cuda", dtype=torch.float16) -> Bundle:
    from transformers import AutoTokenizer
    path = download(name, cache_dir)
    tok = AutoTokenizer.from_pretrained(str(path))
    sd = read_state_dict(path)
    model = FastGPT2.from_hf_state_dict(sd).to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    b = Bundle(name, model, tok, CHECKPOINTS[name]["latent"], tok.eos_token_id, path=path)
    if b.latent:
        ids = tok.convert_tokens_to_ids(list(LATENT_TOKENS))
        if any(i is None or i == tok.unk_token_id or i >= model.vocab_size for i in ids):
            raise ValueError(f"{name}: latent tokens missing from tokenizer: {ids}")
        b.latent_id, b.start_id, b.end_id = ids
    if len(tok) > model.vocab_size:
        raise ValueError(f"{name}: tokenizer has {len(tok)} tokens but embedding has {model.vocab_size}")
    return b


def hf_state_dict(name: str, cache_dir: str | Path) -> tuple[dict, Path]:
    """Raw HF-style state dict (prefixes stripped) for building a trainable HF GPT2 (reward models)."""
    path = download(name, cache_dir)
    sd = read_state_dict(path)
    for prefix in ("base_causallm.", "module.", "model."):
        if any(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}
    return sd, path


def describe(b: Bundle) -> dict:
    return {"name": b.name, "path": str(b.path), "vocab": b.model.vocab_size, "latent": b.latent,
            "start_id": b.start_id, "end_id": b.end_id, "latent_id": b.latent_id, "eos_id": b.eos_id}


def summary_json(b: Bundle) -> dict:
    f = Path(b.path) / "summary.json"
    return json.loads(f.read_text()) if f.exists() else {}
