"""Bộ kiểm thử nhanh — chạy trong notebook 00 (và trước mỗi lần train) để phát hiện lỗi SỚM.

Mỗi kiểm thử in ✅ hoặc ❌ kèm giải thích.  Toàn bộ chạy < 2 phút trên CPU, vài giây trên GPU.
"""
from __future__ import annotations

import math
import random
import tempfile
import traceback
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import data as D
from . import decoding as DEC
from . import metrics as M
from . import models as MD
from .common import normalize_text
from .train import TrainConfig, Trainer

V = 40  # từ vựng giả lập cho kiểm thử


def _tiny_s2s(**kw):
    cfg = MD.Seq2SeqConfig(vocab_size=V, d_model=32, n_heads=4, d_ff=64, enc_layers=2, dec_layers=2,
                           dropout=0.0, attn_dropout=0.0, max_len=128, **kw)
    return MD.TransformerSeq2Seq(cfg)


def _tiny_gpt(**kw):
    cfg = MD.GPTConfig(vocab_size=V, d_model=32, n_heads=4, d_ff=64, n_layers=3, dropout=0.0,
                       attn_dropout=0.0, max_len=128, **kw)
    return MD.GPTDecoder(cfg)


SEP = 4  # id <sep> giả lập


# ---------------------------------------------------------------------------- kiểm thử
def t_normalize():
    a = "Việt Nam  ​ đẹp"   # dạng tổ hợp + khoảng trắng lạ
    assert normalize_text(a) == "Việt Nam đẹp", normalize_text(a)


def t_clean_line():
    assert D.clean_line("It &apos;s a &quot; test &quot; .", "en") == 'It\'s a "test".'
    assert D.clean_line("Tôi nói , &quot; Xin chào &quot; .", "vi") == 'Tôi nói, "Xin chào".'
    assert D.clean_line("   ", "vi") == ""


def t_sampler():
    lengths = np.random.RandomState(0).randint(1, 60, 500)
    s = D.TokenBatchSampler(lengths, max_tokens=300, seed=1)
    b0, b0b, b1 = s.batches(0), s.batches(0), s.batches(1)
    assert b0 == b0b, "batch phải xác định theo epoch (để resume đúng)"
    assert b0 != b1
    flat = sorted(i for b in b0 for i in b)
    assert flat == list(range(500)), "mỗi câu xuất hiện đúng 1 lần"
    for b in b0:
        assert len(b) == 1 or max(lengths[b]) * len(b) <= 300


def t_collate():
    batch = [(0, [5, 6, 7], [8, 9]), (1, [5], [8, 9, 10, 11])]
    g = D.gpt_collate(batch, sep=SEP)
    # câu 0: <s> 5 6 7 <sep> 8 9 </s>  -> nhãn chỉ ở 8 9 </s>
    lab = g["labels"][0].tolist()
    assert lab[:4] == [-100] * 4 and lab[4:7] == [8, 9, D.EOS_ID], lab
    s = D.seq2seq_collate(batch)
    assert s["src"][0].tolist() == [5, 6, 7, D.EOS_ID]
    assert s["tgt_in"][1].tolist() == [D.BOS_ID, 8, 9, 10, 11]
    assert s["tgt_out"][1].tolist() == [8, 9, 10, 11, D.EOS_ID]


def t_attention_sdpa_vs_manual():
    torch.manual_seed(0)
    m = _tiny_s2s().eval()
    src = torch.randint(5, V, (3, 7)); src[0, 5:] = 0
    tgt = torch.randint(5, V, (3, 6)); tgt[1, 4:] = 0
    mem1, ok = m.encode(src)
    mem2, _ = m.encode(src, need_weights=True)
    assert torch.allclose(mem1, mem2, atol=1e-5)
    l1 = m.decode(tgt, mem1, ok)
    l2 = m.decode(tgt, mem2, ok, need_weights=True)
    assert torch.allclose(l1, l2, atol=1e-4)


def t_causality():
    torch.manual_seed(0)
    m = _tiny_s2s().eval()
    src = torch.randint(5, V, (2, 6))
    tgt = torch.randint(5, V, (2, 8))
    a = m(src, tgt)
    tgt2 = tgt.clone(); tgt2[:, 5:] = torch.randint(5, V, (2, 3))
    b = m(src, tgt2)
    assert torch.allclose(a[:, :5], b[:, :5], atol=1e-5), "decoder nhìn trộm tương lai!"
    g = _tiny_gpt().eval()
    x = torch.randint(5, V, (2, 10))
    y = x.clone(); y[:, 6:] = torch.randint(5, V, (2, 4))
    assert torch.allclose(g(x)[:, :6], g(y)[:, :6], atol=1e-5), "GPT nhìn trộm tương lai!"


def t_padding_invariance():
    torch.manual_seed(0)
    m = _tiny_s2s().eval()
    src = torch.randint(5, V, (1, 5)); tgt = torch.randint(5, V, (1, 4))
    src_p = torch.cat([src, torch.zeros(1, 3, dtype=torch.long)], 1)
    assert torch.allclose(m(src, tgt), m(src_p, tgt), atol=1e-5), "đệm nguồn làm thay đổi kết quả"
    g = _tiny_gpt().eval()
    x = torch.randint(5, V, (1, 6))
    xp = torch.cat([torch.zeros(1, 3, dtype=torch.long), x], 1)
    mp = xp != 0
    out = g(xp, mp)
    assert not torch.isnan(out).any(), "NaN khi đệm trái"
    assert torch.allclose(g(x)[:, -1], out[:, -1], atol=1e-5), "đệm trái làm thay đổi kết quả GPT"


def t_beam1_equals_greedy():
    torch.manual_seed(0)
    m = _tiny_s2s().eval()
    src = torch.randint(5, V, (4, 6)); src[:, -1] = D.EOS_ID
    mem, ok = m.encode(src)
    bos = torch.full((4, 1), D.BOS_ID)

    def nl(gen):
        lg = m.decode(torch.cat([bos[:gen.size(0)], gen], 1), mem[:gen.size(0)], ok[:gen.size(0)])
        return torch.log_softmax(lg[:, -1], -1)
    g = DEC._greedy_core(nl, 4, 15, src.device)
    b = DEC._beam_core(nl, 4, 1, 15, src.device)
    assert g == b, f"{g} != {b}"


def t_beam_finds_better_or_equal():
    torch.manual_seed(1)
    m = _tiny_s2s().eval()
    src = torch.randint(5, V, (3, 6)); src[:, -1] = D.EOS_ID
    g = DEC.seq2seq_generate_batch(m, src, beam_size=1)
    b = DEC.seq2seq_generate_batch(m, src, beam_size=5)
    assert len(b) == 3 and all(isinstance(x, list) for x in b)

    def score(i, seq):
        tgt_in = torch.tensor([[D.BOS_ID] + seq])
        tgt_out = torch.tensor([seq + [D.EOS_ID]])
        lp = torch.log_softmax(m(src[i:i + 1], tgt_in), -1)
        return lp.gather(-1, tgt_out.unsqueeze(-1)).sum().item() / len(tgt_out[0])
    for i in range(3):
        assert score(i, b[i]) >= score(i, g[i]) - 1e-4, "beam cho điểm kém hơn greedy"


def t_batch_equals_single():
    torch.manual_seed(0)
    g = _tiny_gpt().eval()
    srcs = [[5, 6, 7, 8, 9], [10, 11], [12, 13, 14]]
    batch = DEC.gpt_generate_batch(g, srcs, SEP, beam_size=1, device=torch.device("cpu"))
    single = [DEC.gpt_generate_batch(g, [s], SEP, beam_size=1, device=torch.device("cpu"))[0] for s in srcs]
    assert batch == single, "GPT: dịch theo batch khác dịch từng câu (lỗi đệm trái / vị trí)"
    bb = DEC.gpt_generate_batch(g, srcs, SEP, beam_size=3, device=torch.device("cpu"))
    bs = [DEC.gpt_generate_batch(g, [s], SEP, beam_size=3, device=torch.device("cpu"))[0] for s in srcs]
    assert bb == bs, "GPT beam: batch khác từng câu"
    m = _tiny_s2s().eval()
    s2 = [(i, s, None) for i, s in enumerate(srcs)]
    b = DEC.seq2seq_generate_batch(m, D.seq2seq_collate(s2)["src"], beam_size=3)
    one = [DEC.seq2seq_generate_batch(m, D.seq2seq_collate([x])["src"], beam_size=3)[0] for x in s2]
    assert b == one, "Seq2seq beam: batch khác từng câu"


def t_label_smoothing():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, V)
    tgt = torch.randint(0, V, (2, 5)); tgt[0, 3:] = -100
    a, _, n = MD.label_smoothed_ce(logits, tgt, 0.1)
    b = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=-100, label_smoothing=0.1)
    assert abs(a.item() - b.item()) < 1e-5 and n == 8, (a.item(), b.item(), n)


def t_bleu_matches_sacrebleu():
    rng = random.Random(0)
    words = "tôi bạn là học sinh giáo viên yêu Việt Nam . , một hai ba".split()
    refs = [" ".join(rng.choice(words) for _ in range(rng.randint(3, 15))) for _ in range(200)]
    hyps = [" ".join(w if rng.random() < .7 else rng.choice(words) for w in r.split()) for r in refs]
    ours = M.bleu_from_stats(M.bleu_stats(hyps, refs).sum(0))
    ref = M.quick_bleu(hyps, refs)
    assert abs(ours - ref) < 0.01, f"{ours} vs {ref}"


def _toy_pairs(n, rng):
    # Bài toán đồ chơi: "dịch" = đảo ngược chuỗi và cộng 1 vào mỗi token
    pairs = []
    for _ in range(n):
        s = [rng.randint(5, V - 2) for _ in range(rng.randint(2, 8))]
        pairs.append((s, [x + 1 for x in reversed(s)]))
    return pairs


def _train_toy(kind, steps, ckpt_dir, max_steps):
    rng = random.Random(0)
    torch.manual_seed(0)
    pairs = _toy_pairs(3000, rng)
    src, tgt = [p[0] for p in pairs], [p[1] for p in pairs]
    ds = D.PairDataset(src, tgt)
    if kind == "s2s":
        model = MD.TransformerSeq2Seq(MD.Seq2SeqConfig(vocab_size=V, d_model=64, n_heads=4, d_ff=128, enc_layers=2,
                                                      dec_layers=2, dropout=0.0, attn_dropout=0.0, max_len=64))
        collate = D.seq2seq_collate
        lengths = [max(len(s), len(t)) + 1 for s, t in pairs]

        def loss_fn(m, b, cfg):
            return MD.label_smoothed_ce(m(b["src"], b["tgt_in"]), b["tgt_out"], cfg.label_smoothing, ignore_index=0)
    else:
        model = MD.GPTDecoder(MD.GPTConfig(vocab_size=V, d_model=64, n_heads=4, d_ff=128, n_layers=3, dropout=0.0,
                                           attn_dropout=0.0, max_len=64))
        collate = partial(D.gpt_collate, sep=SEP)
        lengths = [len(s) + len(t) + 3 for s, t in pairs]

        def loss_fn(m, b, cfg):
            return MD.label_smoothed_ce(m(b["input_ids"], b["attn_mask"]), b["labels"], cfg.label_smoothing)
    test = _toy_pairs(50, random.Random(99))

    def eval_fn(m):
        if kind == "s2s":
            out = DEC.seq2seq_generate_batch(m, D.seq2seq_collate([(i, s, None) for i, (s, _) in enumerate(test)])["src"])
        else:
            out = DEC.gpt_generate_batch(m, [s for s, _ in test], SEP, device=torch.device("cpu"))
        acc = float(np.mean([o == t for o, (_, t) in zip(out, test)]))
        return {"score": 100 * acc}
    cfg = TrainConfig(epochs=100, max_tokens=600, lr=3e-3, warmup_steps=50, eval_every_steps=steps,
                      save_every_steps=steps, log_every_steps=steps, max_steps=max_steps, patience=100,
                      label_smoothing=0.0, amp="fp32")
    tr = Trainer(model, ds, collate, lengths, cfg, ckpt_dir, loss_fn, eval_fn, device=torch.device("cpu"),
                 run_name=f"toy-{kind}")
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        tr.train()
    return tr


def t_train_resume_and_learn():
    # (loại mô hình, tổng số bước, ngưỡng độ chính xác tối thiểu)
    # Mô hình chưa học: độ chính xác (khớp cả chuỗi) ≈ 0%.  Ngưỡng 50% đủ chứng minh mô hình học đúng,
    # đồng thời chừa biên cho sai khác số học giữa các máy/phiên bản PyTorch (đã thấy 76%–92% tuỳ máy).
    for kind, total, thr in (("s2s", 600, 50), ("gpt", 1200, 50)):
        with tempfile.TemporaryDirectory() as d:
            # Chạy 150 bước -> giả lập "mất kết nối" -> chạy lại, phải tiếp tục từ bước 150
            tr1 = _train_toy(kind, 150, d, 150)
            assert tr1.state["global_step"] == 150
            tr2 = _train_toy(kind, 150, d, total)
            assert tr2.state["global_step"] == total, tr2.state["global_step"]
            best = tr2.state["best_score"]
            assert best >= thr, f"{kind}: mô hình không học được bài toán đồ chơi (acc={best:.0f}%)"
            losses = [h["train_loss"] for h in tr2.history if "train_loss" in h]
            assert losses[-1] < 0.5 * losses[0], f"{kind}: loss giảm quá ít ({losses[0]:.2f} -> {losses[-1]:.2f})"
            assert (Path(d) / "best.pt").exists() and (Path(d) / "history.csv").exists()


TESTS = [
    ("Chuẩn hoá Unicode tiếng Việt", t_normalize),
    ("Làm sạch dòng (HTML + detokenize)", t_clean_line),
    ("Batch theo số token, xác định theo epoch", t_sampler),
    ("Ghép batch Seq2Seq / GPT (mặt nạ nhãn)", t_collate),
    ("Attention: SDPA == cài đặt thủ công", t_attention_sdpa_vs_manual),
    ("Mặt nạ nhân quả (không nhìn tương lai)", t_causality),
    ("Bất biến với padding (kể cả đệm trái GPT)", t_padding_invariance),
    ("Beam search K=1 == greedy", t_beam1_equals_greedy),
    ("Beam search không kém greedy", t_beam_finds_better_or_equal),
    ("Dịch theo batch == dịch từng câu", t_batch_equals_single),
    ("Label smoothing CE == PyTorch", t_label_smoothing),
    ("BLEU tự tính == sacreBLEU", t_bleu_matches_sacrebleu),
    ("Huấn luyện + resume + học được bài toán đồ chơi", t_train_resume_and_learn),
]


def run_all(skip_slow=False):
    ok = True
    for name, fn in TESTS:
        if skip_slow and fn is t_train_resume_and_learn:
            continue
        try:
            fn()
            print(f"✅ {name}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"❌ {name}: {e}")
            traceback.print_exc()
    print("\n" + ("🎉 TẤT CẢ KIỂM THỬ ĐỀU QUA" if ok else "⛔ CÓ KIỂM THỬ THẤT BẠI — KHÔNG chạy huấn luyện cho tới khi sửa xong"))
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
