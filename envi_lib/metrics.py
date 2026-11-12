"""Các độ đo đánh giá chất lượng dịch máy (phần chạy trong môi trường chính).

  * BLEU, chrF++, TER  : sacreBLEU (chuẩn hoá cách token hoá => điểm so sánh được giữa các bài báo)
  * METEOR             : NLTK (lưu ý: phần đồng nghĩa WordNet chỉ có cho tiếng Anh)
  * Kiểm định ý nghĩa thống kê: paired bootstrap resampling (Koehn, 2004)
  * Phân tích theo độ dài câu

COMET và BERTScore (độ đo dựa trên mạng nơ-ron) chạy trong môi trường Python riêng — xem eval_neural.py.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

MAX_N = 4


def corpus_scores(hyps, refs) -> dict:
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF, TER
    assert len(hyps) == len(refs), f"{len(hyps)} bản dịch != {len(refs)} tham chiếu"
    bleu, chrf, ter = BLEU(), CHRF(word_order=2), TER()
    b = bleu.corpus_score(hyps, [refs])
    c = chrf.corpus_score(hyps, [refs])
    t = ter.corpus_score(hyps, [refs])
    return {
        "BLEU": round(b.score, 2), "chrF++": round(c.score, 2), "TER": round(t.score, 2),
        "BP": round(b.bp, 3), "len_ratio": round(b.sys_len / max(1, b.ref_len), 3),
        "p1": round(b.precisions[0], 1), "p2": round(b.precisions[1], 1),
        "p3": round(b.precisions[2], 1), "p4": round(b.precisions[3], 1),
        "_sig_bleu": str(bleu.get_signature()), "_sig_chrf": str(chrf.get_signature()),
        "_sig_ter": str(ter.get_signature()), "_sacrebleu": sacrebleu.__version__,
    }


def quick_bleu(hyps, refs) -> float:
    from sacrebleu.metrics import BLEU
    return BLEU().corpus_score(hyps, [refs]).score


# ----------------------------------------------------------------------------
# BLEU theo thống kê từng câu (để bootstrap nhanh) — kiểm chứng khớp sacreBLEU
# ----------------------------------------------------------------------------

def _tok13a():
    from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
    return Tokenizer13a()


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def bleu_stats(hyps, refs) -> np.ndarray:
    """Mỗi câu: [len_hyp, len_ref, match_1..4, total_1..4]."""
    tok = _tok13a()
    out = np.zeros((len(hyps), 2 + 2 * MAX_N), dtype=np.int64)
    for i, (h, r) in enumerate(zip(hyps, refs)):
        ht, rt = tok(h.strip()).split(), tok(r.strip()).split()
        out[i, 0], out[i, 1] = len(ht), len(rt)
        for n in range(1, MAX_N + 1):
            hc, rc = _ngrams(ht, n), _ngrams(rt, n)
            out[i, 1 + n] = sum(min(c, rc[g]) for g, c in hc.items())
            out[i, 1 + MAX_N + n] = max(0, len(ht) - n + 1)
    return out


def bleu_from_stats(s) -> float:
    """Giống sacreBLEU mặc định (smooth='exp')."""
    sys_len, ref_len = s[0], s[1]
    if sys_len == 0:
        return 0.0
    log_sum, smooth = 0.0, 1.0
    for n in range(MAX_N):
        m, t = s[2 + n], s[2 + MAX_N + n]
        if t == 0:
            return 0.0
        if m == 0:
            smooth *= 2
            p = 1.0 / (smooth * t)
        else:
            p = m / t
        log_sum += math.log(p)
    bp = 1.0 if sys_len > ref_len else math.exp(1 - ref_len / sys_len)
    return 100 * bp * math.exp(log_sum / MAX_N)


def paired_bootstrap(stats_base, stats_sys, n_samples=1000, seed=12345):
    """So sánh hệ thống 'sys' với 'base' trên cùng tập test.

    Trả về: delta BLEU, khoảng tin cậy 95% của mỗi hệ thống, p-value (xác suất sys KHÔNG tốt hơn base).
    """
    rng = np.random.RandomState(seed)
    N = len(stats_base)
    bb, bs = [], []
    for _ in range(n_samples):
        idx = rng.randint(0, N, N)
        bb.append(bleu_from_stats(stats_base[idx].sum(0)))
        bs.append(bleu_from_stats(stats_sys[idx].sum(0)))
    bb, bs = np.array(bb), np.array(bs)
    delta = bs - bb
    return {
        "base_bleu": round(bleu_from_stats(stats_base.sum(0)), 2),
        "sys_bleu": round(bleu_from_stats(stats_sys.sum(0)), 2),
        "delta": round(float(delta.mean()), 2),
        "base_ci95": (round(float(np.percentile(bb, 2.5)), 2), round(float(np.percentile(bb, 97.5)), 2)),
        "sys_ci95": (round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)),
        "p_value": round(float((delta <= 0).mean()), 4),
    }


def bootstrap_mean_ci(values, n_samples=1000, seed=12345):
    rng = np.random.RandomState(seed)
    v = np.asarray(values, dtype=float)
    means = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(n_samples)]
    return float(v.mean()), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def paired_bootstrap_mean(base_vals, sys_vals, n_samples=1000, seed=12345):
    rng = np.random.RandomState(seed)
    a, b = np.asarray(base_vals, float), np.asarray(sys_vals, float)
    d = []
    for _ in range(n_samples):
        idx = rng.randint(0, len(a), len(a))
        d.append(b[idx].mean() - a[idx].mean())
    d = np.array(d)
    return {"delta": float(b.mean() - a.mean()), "p_value": float((d <= 0).mean())}


# ----------------------------------------------------------------------------
# METEOR
# ----------------------------------------------------------------------------

_METEOR_OK = None


def _meteor_ready() -> bool:
    """Tải WordNet MỘT lần; nếu không được thì báo một lần và bỏ qua METEOR."""
    global _METEOR_OK
    if _METEOR_OK is None:
        # Lưu ý: nltk.download('wordnet') chỉ tải file .zip (không giải nén) => nltk.data.find('corpora/wordnet')
        # báo không thấy dù vẫn dùng được.  Vì vậy kiểm tra bằng cách DÙNG THỬ WordNet.
        def usable():
            from nltk.corpus import wordnet
            wordnet.ensure_loaded()
            return len(wordnet.synsets("dog")) > 0
        try:
            import nltk
            try:
                ok = usable()
            except LookupError:
                for res in ("wordnet", "omw-1.4"):
                    nltk.download(res, quiet=True)
                ok = usable()
            _METEOR_OK = bool(ok)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Không tải được WordNet cho METEOR — bỏ qua METEOR ({type(e).__name__}).")
            _METEOR_OK = False
    return _METEOR_OK


def meteor(hyps, refs):
    """METEOR trung bình theo câu (NLTK).  Trả về (None, None) nếu không có WordNet."""
    if not _meteor_ready():
        return None, None
    from nltk.translate.meteor_score import meteor_score
    scores = [meteor_score([r.lower().split()], h.lower().split()) for h, r in zip(hyps, refs)]
    return round(100 * float(np.mean(scores)), 2), scores


# ----------------------------------------------------------------------------
# Phân tích theo độ dài
# ----------------------------------------------------------------------------

LENGTH_BINS = [(1, 10), (11, 20), (21, 30), (31, 50), (51, 10_000)]


def bleu_by_length(hyps, refs, srcs, bins=LENGTH_BINS):
    """BLEU theo nhóm độ dài câu NGUỒN (số từ tiếng Anh)."""
    from sacrebleu.metrics import BLEU
    out = {}
    lens = [len(s.split()) for s in srcs]
    for lo, hi in bins:
        idx = [i for i, L in enumerate(lens) if lo <= L <= hi]
        label = f"{lo}-{hi}" if hi < 10_000 else f">{lo - 1}"
        if not idx:
            continue
        out[label] = {"n": len(idx), "BLEU": round(BLEU().corpus_score([hyps[i] for i in idx],
                                                                       [[refs[i] for i in idx]]).score, 2)}
    return out
