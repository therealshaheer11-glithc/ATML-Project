"""GSM8k-aug data (same source and conversion as the official Coconut repo) and answer parsing."""
from __future__ import annotations

import json
import random
from pathlib import Path
from urllib.request import urlopen

REVISION = "e06a32ee5e4cd117171daeb4755d2a97ece62761"
URL = "https://{host}/da03/Internalize_CoT_Step_by_Step/{rev}/data/gsm8k/{split}.txt"
EXPECTED = {"train": 385_620, "valid": 500, "test": 1_319}


def _fetch_lines(url, expected, tries=4):
    for t in range(tries):
        try:
            with urlopen(url, timeout=600) as r:
                lines = [l for l in r.read().decode("utf-8").split("\n") if l.strip()]  # not splitlines(): some questions contain U+2028 etc.
            if len(lines) == expected and all("||" in l and "##" in l for l in lines):
                return lines
            print(f"[data] incomplete download ({len(lines)}/{expected} lines), retry {t + 1}")
        except OSError as e:
            print(f"[data] download error {e}, retry {t + 1}")
    raise RuntimeError(f"could not download {url}")


def download(data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in ("test", "valid", "train"):
        out = data_dir / f"gsm_{split}.json"
        if out.exists():
            continue
        host = "media.githubusercontent.com/media" if split == "train" else "raw.githubusercontent.com"
        lines = _fetch_lines(URL.format(host=host, rev=REVISION, split=split), EXPECTED[split])
        rows = []
        for line in lines:
            # identical to coconut/preprocessing/gsm_icot.py
            rows.append({"question": line.split("||")[0],
                         "steps": line.split("||")[1].split("##")[0].strip().split(" "),
                         "answer": line.split("##")[-1].strip()})
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows))
        tmp.rename(out)
        print(f"[data] {split}: {len(rows)} rows")


def load(data_dir: str | Path, split: str) -> list[dict]:
    rows = json.loads((Path(data_dir) / f"gsm_{split}.json").read_text())
    for i, r in enumerate(rows):
        r["qid"] = i
        r["gold"] = normalize(r["answer"])
    return rows


def train_subset(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Fixed random subset (any prefix of it is itself a random subset)."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    return [rows[i] for i in idx[:n]]


def normalize(s: str) -> str:
    return str(s).replace(",", "").strip()


def parse_answer(generated_text: str) -> str:
    """Official Coconut rule (run.py): text.split('#')[-1].replace(',', '').strip().
    Applied to the generated continuation only. Empty string if no '#'."""
    if "#" not in generated_text:
        return ""
    return normalize(generated_text.split("#")[-1])


def numeric_equal(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-9
    except ValueError:
        return False


def prompt_ids(tok, question: str) -> list[int]:
    # Coconut / LatentITS: question + "\n", no special tokens
    return tok.encode(question + "\n", add_special_tokens=False)
