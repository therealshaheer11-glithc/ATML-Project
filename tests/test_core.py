"""CPU tests: FastGPT2 == HF GPT-2; batched Coconut sampler == naive official-style loop;
replay is exact; pass@N and selectors match brute force.   Run:  python -m pytest -q tests"""
import itertools
import math

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from cb import stats as S
from cb.gpt2 import Drop, FastGPT2
from cb.models import Bundle
from cb.sampling import coconut_rollout, text_rollout

torch.manual_seed(0)
V, EOS, START, END, LAT = 60, 0, 57, 58, 59


def hf_model(std=0.3):
    cfg = GPT2Config(vocab_size=V, n_positions=128, n_embd=64, n_layer=3, n_head=4)
    m = GPT2LMHeadModel(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, std)
    return m


class Tok:
    def encode(self, s, add_special_tokens=False):
        return [1 + (ord(c) % 50) for c in s]

    def decode(self, ids, skip_special_tokens=True):
        return f"### {sum(ids) % 10}" if ids else ""


def bundle(hf):
    fm = FastGPT2.from_hf_state_dict(hf.state_dict(), n_head=4).eval()
    return Bundle("t", fm, Tok(), True, EOS, START, END, LAT)


def test_prefill_and_cache_match_hf():
    hf = hf_model(); b = bundle(hf); m = b.model
    ids = torch.randint(1, 50, (3, 9))
    mask = torch.ones(3, 9, dtype=torch.long); mask[0, :4] = 0; mask[1, :2] = 0   # left padding
    pos = (mask.cumsum(1) - 1).clamp_min(0)
    ref = hf.transformer(input_ids=ids, attention_mask=mask, position_ids=pos).last_hidden_state
    cache = m.new_cache(3, 20)
    out = m(m.embed(ids), cache, mask)
    assert torch.allclose(out[mask.bool()], ref[mask.bool()], atol=1e-4)
    # incremental step == full recompute
    nxt = torch.randint(1, 50, (3, 1))
    step = m(m.embed(nxt), cache)
    ids2 = torch.cat([ids, nxt], 1); mask2 = torch.cat([mask, torch.ones(3, 1, dtype=torch.long)], 1)
    ref2 = hf.transformer(input_ids=ids2, attention_mask=mask2,
                          position_ids=(mask2.cumsum(1) - 1).clamp_min(0)).last_hidden_state
    assert torch.allclose(step[:, 0], ref2[:, -1], atol=1e-4)


def naive_coconut(hf, prompt, max_new=8):
    """Official Coconut style: recompute the full sequence each step with inputs_embeds."""
    E = hf.transformer.wte
    x = E(torch.tensor([prompt + [START]]))
    thoughts = []
    for _ in range(6):
        h = hf.transformer(inputs_embeds=x).last_hidden_state[:, -1:]
        thoughts.append(h[0, 0]); x = torch.cat([x, h], 1)
    x = torch.cat([x, E(torch.tensor([[END]]))], 1)
    out = []
    for _ in range(max_new):
        t = hf(inputs_embeds=x).logits[0, -1].argmax().item()
        out.append(t)
        if t == EOS:
            break
        x = torch.cat([x, E(torch.tensor([[t]]))], 1)
    return torch.stack(thoughts), out


def test_coconut_matches_naive_and_batches():
    hf = hf_model(); b = bundle(hf)
    prompts = [[3, 4, 5, 6, 7], [8, 9], [10, 11, 12]]
    res = coconut_rollout(b, prompts, n=2, p=0.0, gen=None, max_new=8)
    for q, pr in enumerate(prompts):
        th, ans = naive_coconut(hf, pr)
        for j in range(2):
            r = q * 2 + j
            assert torch.allclose(res["thoughts"][r], th, atol=1e-3)
            got = [t for t in res["ids"][r].tolist() if t >= 0]
            assert got == ans[:len(got)] and len(got) == len(ans)


def test_dropout_replay_and_prefix():
    hf = hf_model(); b = bundle(hf)
    prompts = [[3, 4, 5], [6, 7, 8, 9]]
    g1 = torch.Generator().manual_seed(7); g2 = torch.Generator().manual_seed(7)
    a = coconut_rollout(b, prompts, 4, 0.2, g1)
    c = coconut_rollout(b, prompts, 4, 0.2, g2)
    assert torch.equal(a["thoughts"], c["thoughts"])                 # exact replay
    assert not torch.allclose(a["thoughts"][0], a["thoughts"][1])     # samples differ
    # all 6 thoughts fixed: continuation is deterministic (re-encoded with dropout off, as LatentITS)
    pre = a["thoughts"][[0, 4]]
    d = coconut_rollout(b, prompts, 1, 0.2, torch.Generator().manual_seed(1), prefix=pre)
    d2 = coconut_rollout(b, prompts, 1, 0.2, torch.Generator().manual_seed(5), prefix=pre)
    assert torch.equal(d["ids"], d2["ids"])
    # with p=0 the original path re-decoded from its own thoughts gives the same answer
    z = coconut_rollout(b, prompts, 1, 0.0, None)
    z2 = coconut_rollout(b, prompts, 1, 0.0, None, prefix=z["thoughts"])
    assert torch.equal(z["ids"], z2["ids"])
    # prefix of 3: first 3 thoughts kept, rest resampled
    e = coconut_rollout(b, prompts, 3, 0.2, torch.Generator().manual_seed(2), prefix=a["thoughts"][[0, 4], :3])
    assert torch.allclose(e["thoughts"][0, :3], a["thoughts"][0, :3])


def test_hf_dropout_sites_statistically():
    """Mean output with our dropout ~= HF train-mode mean (same sites)."""
    hf = hf_model(); b = bundle(hf); m = b.model
    for mod in hf.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.1
    ids = torch.randint(1, 50, (1, 6))
    hf.train()
    ref = torch.stack([hf.transformer(input_ids=ids).last_hidden_state[0, -1] for _ in range(400)]).mean(0)
    hf.eval()
    g = torch.Generator().manual_seed(0)
    ours = []
    for _ in range(400):
        c = m.new_cache(1, 8)
        ours.append(m(m.embed(ids), c, drop=Drop(0.1, g))[0, -1])
    ours = torch.stack(ours).mean(0)
    det = hf.transformer(input_ids=ids).last_hidden_state[0, -1]
    assert (ours - ref).norm() < 0.5 * (det - ref).norm() + 0.15


def test_text_greedy_matches_hf_generate():
    hf = hf_model(); b = bundle(hf)
    pr = [3, 4, 5, 6]
    r = text_rollout(b, [pr], 1, 0.0, None, max_new=6)
    g = hf.generate(torch.tensor([pr]), max_new_tokens=6, do_sample=False, eos_token_id=EOS, pad_token_id=EOS)
    want = g[0, len(pr):].tolist()
    got = [t for t in r["ids"][0].tolist() if t >= 0]
    assert got == want[:len(got)]


def test_pass_at_n_and_selectors():
    rng = np.random.default_rng(0)
    n = 8
    cor = rng.random((50, n)) < 0.3
    for N in (1, 2, 4, 8):
        brute = np.array([np.mean([cor[i, list(c)].any() for c in itertools.combinations(range(n), N)])
                          for i in range(50)])
        assert np.allclose(S.pass_at_n(cor, N), brute)
    # majority on a hand-made question: answers a a b c, correct = b -> majority picks a
    codes = np.array([[0, 0, 1, 2]]); cr = np.array([[False, False, True, False]])
    cc = [np.array([False, True, False])]
    acc, one, tie, distinct = S.selectors(codes, cr, cc, {"s": np.array([[0.1, 0.2, 0.9, 0.3]])}, 4, rng,
                                          best_of=["s"], vote=["s"])
    assert acc["majority"][0] == 0 and acc["best:s"][0] == 1 and acc["vote:s"][0] == 1  # sums a=.3 b=.9 c=.3
    assert math.isclose(acc["random"][0], 0.25)


def test_within_question_auroc():
    cor = np.array([[True, False, False], [True, True, True]])
    sc = np.array([[0.9, 0.1, 0.2], [0.1, 0.2, 0.3]])
    v, nq = S.within_question_auroc(sc, cor)
    assert v == 1.0 and nq == 1
