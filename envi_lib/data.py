"""Tải, làm sạch, thống kê dữ liệu IWSLT'15 En-Vi; tokenizer SentencePiece; batching.

Nguồn dữ liệu (theo thứ tự ưu tiên):
  1. File đã có sẵn trong data_raw/stanford/ (người dùng có thể tự tải và đặt vào).
  2. Tải từ máy chủ Stanford NLP: train / tst2012 (dev) / tst2013 (test) — split chuẩn trong các bài báo.
  3. Dự phòng: bản parquet trên Hugging Face (IWSLT/mt_eng_vietnamese).
     LƯU Ý: bộ này có validation == test == tst2013, nên KHÔNG dùng validation của nó;
     tập dev được tách ngẫu nhiên 1.500 câu từ train (cố định seed).
"""
from __future__ import annotations

import html
import os
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .common import Paths, load_json, normalize_text, read_lines, save_json, write_lines

STANFORD_BASE = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/"
STANFORD_SPLITS = {"train": "train", "dev": "tst2012", "test": "tst2013"}
HF_REPO = "IWSLT/mt_eng_vietnamese"
HF_REVISION = "refs/convert/parquet"
HF_FILES = {"train": "iwslt2015-en-vi/train/0000.parquet",
            "test": "iwslt2015-en-vi/test/0000.parquet"}
# Số dòng tham khảo của bản Stanford (dùng để cảnh báo nếu file tải về bị thiếu/hỏng)
EXPECTED_LINES = {"train": 133317, "dev": 1553, "test": 1268}
SRC, TGT = "en", "vi"


# ----------------------------------------------------------------------------
# 1. Tải dữ liệu
# ----------------------------------------------------------------------------

def _download(url: str, dest: Path, retries: int = 3, timeout: int = 60):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (envi-nlp-course)"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if part.stat().st_size == 0:
                raise IOError("file rỗng")
            os.replace(part, dest)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"   lần {attempt}/{retries} lỗi khi tải {url}: {e}")
            time.sleep(2 * attempt)
    if part.exists():
        part.unlink()
    raise IOError(f"Không tải được {url}: {last_err}")


def _stanford_complete(d: Path) -> bool:
    for name in STANFORD_SPLITS.values():
        for lang in (SRC, TGT):
            p = d / f"{name}.{lang}"
            if not p.exists() or p.stat().st_size == 0:
                return False
    return True


def fetch_raw(paths: Paths, source: str = "auto") -> dict:
    """Bảo đảm có dữ liệu thô.  Trả về dict mô tả nguồn đã dùng.

    source: "auto" (Stanford rồi dự phòng HF) | "stanford" | "hf"
    """
    sdir = paths.data_raw / "stanford"
    hdir = paths.data_raw / "hf"
    if source in ("auto", "stanford"):
        if _stanford_complete(sdir):
            print("✔ Đã có dữ liệu Stanford trong", sdir)
            return {"source": "stanford", "dir": str(sdir), "dev_from_train": False}
        try:
            print("⬇ Tải IWSLT'15 En-Vi từ Stanford NLP ...")
            for name in STANFORD_SPLITS.values():
                for lang in (SRC, TGT):
                    p = sdir / f"{name}.{lang}"
                    if not p.exists() or p.stat().st_size == 0:
                        _download(STANFORD_BASE + f"{name}.{lang}", p)
            if _stanford_complete(sdir):
                return {"source": "stanford", "dir": str(sdir), "dev_from_train": False}
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Không tải được từ Stanford ({e}).")
            if source == "stanford":
                raise
    # Dự phòng Hugging Face
    print("⬇ Dùng nguồn dự phòng Hugging Face (parquet) ...")
    import pandas as pd
    from huggingface_hub import hf_hub_download
    hdir.mkdir(parents=True, exist_ok=True)
    for split, fname in HF_FILES.items():
        if (hdir / f"{split}.{SRC}").exists() and (hdir / f"{split}.{TGT}").exists():
            continue
        local = hf_hub_download(HF_REPO, fname, repo_type="dataset", revision=HF_REVISION)
        df = pd.read_parquet(local)
        tr = df["translation"].tolist()
        write_lines(hdir / f"{split}.{SRC}", [t[SRC] for t in tr])
        write_lines(hdir / f"{split}.{TGT}", [t[TGT] for t in tr])
    print("ℹ️  Nguồn HF: tập dev sẽ được tách từ train (vì validation của HF trùng test).")
    return {"source": "hf", "dir": str(hdir), "dev_from_train": True}


def load_raw(info: dict) -> dict:
    d = Path(info["dir"])
    out = {}
    names = STANFORD_SPLITS if info["source"] == "stanford" else {"train": "train", "test": "test"}
    for split, name in names.items():
        src = read_lines(d / f"{name}.{SRC}")
        tgt = read_lines(d / f"{name}.{TGT}")
        # Bỏ các dòng trống ở CUỐI file (thường gặp), rồi mới so số dòng
        while src and tgt and not src[-1].strip() and not tgt[-1].strip():
            src.pop(); tgt.pop()
        if len(src) != len(tgt):
            raise ValueError(f"[{split}] số dòng en ({len(src)}) != vi ({len(tgt)}): file hỏng, hãy xoá {d} và tải lại.")
        exp = EXPECTED_LINES.get(split)
        if exp and info["source"] == "stanford" and abs(len(src) - exp) > 5:
            print(f"⚠️  [{split}] có {len(src)} dòng, khác số tham khảo {exp}. Kiểm tra lại file.")
        out[split] = (src, tgt)
    return out


# ----------------------------------------------------------------------------
# 2. Làm sạch
# ----------------------------------------------------------------------------

_DETOK = {}


def _detok(lang):
    if lang not in _DETOK:
        from sacremoses import MosesDetokenizer
        _DETOK[lang] = MosesDetokenizer(lang=lang)
    return _DETOK[lang]


def clean_line(s: str, lang: str) -> str:
    """Bản Stanford đã được tách từ kiểu Moses và mã hoá HTML (&apos; &quot; ...).

    Ta khôi phục văn bản tự nhiên: giải mã HTML -> ghép lại token (detokenize) -> chuẩn hoá.
    Nhờ vậy, mô hình train từ đầu và mô hình pretrained được đánh giá trên CÙNG một dạng văn bản,
    và sacreBLEU (tự tokenize theo chuẩn 13a) cho điểm so sánh được.
    """
    s = s.strip()
    for _ in range(2):  # một số dòng bị mã hoá 2 lần, ví dụ &amp;apos;
        u = html.unescape(s)
        if u == s:
            break
        s = u
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    if s:
        s = _detok(lang).detokenize(s.split())
    return normalize_text(s)


def clean_split(src, tgt, *, is_train: bool, max_words: int = 150, max_ratio: float = 3.5,
                ratio_min_len: int = 4, block_src: set | None = None):
    """Làm sạch một split.  Trả về (src, tgt, báo_cáo_số_câu_bị_loại_theo_lý_do).

    Tập dev/test chỉ loại cặp rỗng (giữ nguyên để so sánh được với các công trình khác).
    Tập train còn lọc thêm: quá dài, tỉ lệ độ dài bất thường, trùng lặp, trùng với dev/test.
    """
    report = Counter()
    out_s, out_t = [], []
    seen = set()
    for s, t in zip(src, tgt):
        s, t = clean_line(s, SRC), clean_line(t, TGT)
        if not s or not t:
            report["rỗng"] += 1
            continue
        if is_train:
            ls, lt = len(s.split()), len(t.split())
            if ls > max_words or lt > max_words:
                report[f"dài > {max_words} từ"] += 1
                continue
            if min(ls, lt) >= ratio_min_len and max(ls, lt) / min(ls, lt) > max_ratio:
                report[f"tỉ lệ độ dài > {max_ratio}"] += 1
                continue
            if (s, t) in seen:
                report["trùng lặp"] += 1
                continue
            if block_src and s in block_src:
                report["trùng câu nguồn dev/test"] += 1
                continue
            seen.add((s, t))
        out_s.append(s)
        out_t.append(t)
    report["giữ lại"] = len(out_s)
    return out_s, out_t, dict(report)


def build_clean_dataset(paths: Paths, raw_info: dict, dev_size_if_carved: int = 1500, seed: int = 42,
                        **filter_kw) -> dict:
    raw = load_raw(raw_info)
    test_s, test_t, rep_test = clean_split(*raw["test"], is_train=False)
    if raw_info["dev_from_train"]:
        dev_s, dev_t, rep_dev = [], [], {}
    else:
        dev_s, dev_t, rep_dev = clean_split(*raw["dev"], is_train=False)
    block = set(test_s) | set(dev_s)
    tr_s, tr_t, rep_train = clean_split(*raw["train"], is_train=True, block_src=block, **filter_kw)

    if raw_info["dev_from_train"]:
        rng = random.Random(seed)
        idx = list(range(len(tr_s)))
        rng.shuffle(idx)
        dev_idx = set(idx[:dev_size_if_carved])
        dev_s = [tr_s[i] for i in sorted(dev_idx)]
        dev_t = [tr_t[i] for i in sorted(dev_idx)]
        dev_src_set = set(dev_s)
        keep = [i for i in range(len(tr_s)) if i not in dev_idx and tr_s[i] not in dev_src_set]
        tr_s = [tr_s[i] for i in keep]
        tr_t = [tr_t[i] for i in keep]
        rep_dev = {"tách từ train": len(dev_s)}

    # Kiểm tra rò rỉ dữ liệu
    check_no_leakage(tr_s, dev_s, test_s, dev_t, test_t)

    for split, (s, t) in {"train": (tr_s, tr_t), "dev": (dev_s, dev_t), "test": (test_s, test_t)}.items():
        write_lines(paths.data_clean / f"{split}.{SRC}", s)
        write_lines(paths.data_clean / f"{split}.{TGT}", t)
    info = {
        "source": raw_info["source"],
        "dev_from_train": raw_info["dev_from_train"],
        "dev_name": "tst2012" if not raw_info["dev_from_train"] else f"{dev_size_if_carved} câu tách từ train",
        "test_name": "tst2013",
        "counts": {"train": len(tr_s), "dev": len(dev_s), "test": len(test_s)},
        "clean_report": {"train": rep_train, "dev": rep_dev, "test": rep_test},
        "filters": {"max_words": filter_kw.get("max_words", 150), "max_ratio": filter_kw.get("max_ratio", 3.5)},
    }
    save_json(info, paths.data_clean / "data_info.json")
    return info


def check_no_leakage(tr_s, dev_s, test_s, dev_t, test_t, max_overlap: float = 0.05):
    test_pairs = set(zip(test_s, test_t))
    dev_pairs = set(zip(dev_s, dev_t))
    if test_pairs and dev_pairs:
        ov = len(test_pairs & dev_pairs) / len(test_pairs)
        if ov > max_overlap:
            raise RuntimeError(f"❌ Dev và test trùng {ov:.0%} — chọn checkpoint trên dev sẽ làm sai lệch kết quả test!")
    tr = set(tr_s)
    n_tr_test = sum(1 for s in test_s if s in tr)
    if n_tr_test:
        raise RuntimeError(f"❌ {n_tr_test} câu nguồn của test còn nằm trong train.")
    print(f"✔ Kiểm tra rò rỉ: dev∩test = {len(test_pairs & dev_pairs)} cặp, train∩test(nguồn) = 0")


def load_clean(paths: Paths, split: str):
    s = read_lines(paths.data_clean / f"{split}.{SRC}")
    t = read_lines(paths.data_clean / f"{split}.{TGT}")
    if len(s) != len(t):
        raise ValueError(f"data_clean/{split}: số dòng en/vi lệch nhau. Chạy lại notebook 01.")
    return s, t


def load_all_clean(paths: Paths, dry_run: bool = False, dry_sizes=(3000, 200, 200)):
    """Đọc train/dev/test đã làm sạch.  DRY_RUN: lấy tập con nhỏ để chạy thử nhanh."""
    if not (paths.data_clean / "data_info.json").exists():
        raise FileNotFoundError("Chưa có dữ liệu sạch. Hãy chạy notebook 01 trước.")
    data = {sp: load_clean(paths, sp) for sp in ("train", "dev", "test")}
    if dry_run:
        for sp, n in zip(("train", "dev", "test"), dry_sizes):
            s, t = data[sp]
            data[sp] = (s[:n], t[:n])
    return data


def length_stats(lines) -> dict:
    lens = np.array([len(x.split()) for x in lines])
    if len(lens) == 0:
        return {}
    return {"n": int(len(lens)), "mean": float(lens.mean()), "median": float(np.median(lens)),
            "p95": float(np.percentile(lens, 95)), "max": int(lens.max()), "min": int(lens.min())}


# ----------------------------------------------------------------------------
# 3. Tokenizer SentencePiece (dùng chung cho 2 mô hình train từ đầu)
# ----------------------------------------------------------------------------

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SEP_PIECE = "<sep>"


def train_spm(paths: Paths, train_src, train_tgt, vocab_size: int = 16000, seed: int = 42,
              max_sentences: int = 1_000_000):
    """Huấn luyện BPE chung (joint) cho tiếng Anh + tiếng Việt.

    - byte_fallback: ký tự lạ được tách thành byte => không bao giờ sinh <unk>.
    - normalization 'identity': văn bản đã chuẩn hoá NFC ở bước làm sạch; không để SentencePiece
      đổi ký tự (NFKC) khiến bản dịch khác tham chiếu.
    - <sep>: token phân cách nguồn/đích cho mô hình GPT (decoder-only).
    """
    import sentencepiece as spm
    paths.spm.mkdir(parents=True, exist_ok=True)
    lines = list(train_src) + list(train_tgt)
    random.Random(seed).shuffle(lines)
    inp = paths.spm / "spm_input.txt"
    write_lines(inp, lines[:max_sentences])
    spm.SentencePieceTrainer.train(
        input=str(inp), model_prefix=str(paths.spm / "spm_envi"), vocab_size=vocab_size,
        model_type="bpe", character_coverage=0.9995, byte_fallback=True,
        pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
        user_defined_symbols=[SEP_PIECE], normalization_rule_name="identity",
        shuffle_input_sentence=True, input_sentence_size=max_sentences, seed_sentencepiece_size=1_000_000,
        num_threads=max(1, os.cpu_count() or 1), hard_vocab_limit=False, minloglevel=2,
    )
    inp.unlink(missing_ok=True)
    return SpmTokenizer(paths.spm_model)


class SpmTokenizer:
    def __init__(self, model_path):
        import sentencepiece as spm
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Không thấy tokenizer {model_path}. Chạy notebook 01 trước "
                                    "(chú ý DRY_RUN của notebook 01 phải giống notebook hiện tại).")
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.pad_id, self.unk_id, self.bos_id, self.eos_id = PAD_ID, UNK_ID, BOS_ID, EOS_ID
        self.sep_id = self.sp.piece_to_id(SEP_PIECE)
        assert self.sp.pad_id() == PAD_ID and self.sp.eos_id() == EOS_ID and self.sp.bos_id() == BOS_ID
        assert self.sep_id != UNK_ID, "Tokenizer thiếu token <sep>"
        self.special = {PAD_ID, BOS_ID, EOS_ID, self.sep_id}
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, texts):
        return self.sp.encode(list(texts), out_type=int)

    def decode(self, ids_list):
        clean = []
        for ids in ids_list:
            if torch.is_tensor(ids):
                ids = ids.tolist()
            out = []
            for i in ids:
                if i == EOS_ID:
                    break
                if i not in self.special:
                    out.append(int(i))
            clean.append(out)
        return [normalize_text(x) for x in self.sp.decode(clean)]


# ----------------------------------------------------------------------------
# 4. Dataset, collate, batching theo số token
# ----------------------------------------------------------------------------

class PairDataset(torch.utils.data.Dataset):
    def __init__(self, src_ids, tgt_ids=None):
        self.src = src_ids
        self.tgt = tgt_ids

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return i, self.src[i], (self.tgt[i] if self.tgt is not None else None)


def filter_by_token_len(src_ids, tgt_ids, max_src: int, max_tgt: int):
    keep = [i for i, (s, t) in enumerate(zip(src_ids, tgt_ids)) if 0 < len(s) <= max_src and 0 < len(t) <= max_tgt]
    return [src_ids[i] for i in keep], [tgt_ids[i] for i in keep], len(src_ids) - len(keep)


def _pad(seqs, pad, left=False):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s):
            if left:
                out[i, L - len(s):] = torch.as_tensor(s, dtype=torch.long)
            else:
                out[i, :len(s)] = torch.as_tensor(s, dtype=torch.long)
    return out


def seq2seq_collate(batch, pad=PAD_ID, bos=BOS_ID, eos=EOS_ID):
    idx = [b[0] for b in batch]
    src = _pad([list(b[1]) + [eos] for b in batch], pad)
    out = {"idx": torch.tensor(idx), "src": src}
    if batch[0][2] is not None:
        out["tgt_in"] = _pad([[bos] + list(b[2]) for b in batch], pad)
        out["tgt_out"] = _pad([list(b[2]) + [eos] for b in batch], pad)
    return out


IGNORE = -100


def gpt_collate(batch, pad=PAD_ID, bos=BOS_ID, sep=None, eos=EOS_ID):
    """Chuỗi huấn luyện của GPT: <s> nguồn <sep> đích </s>.  Chỉ tính loss trên phần đích."""
    assert sep is not None
    inputs, labels = [], []
    for _, s, t in batch:
        seq = [bos] + list(s) + [sep] + list(t) + [eos]
        inp, lab = seq[:-1], seq[1:]
        lab = [IGNORE] * (len(s) + 1) + lab[len(s) + 1:]
        inputs.append(inp)
        labels.append(lab)
    ids = _pad(inputs, pad)
    lab = _pad(labels, IGNORE)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, x in enumerate(inputs):
        mask[i, :len(x)] = True
    return {"idx": torch.tensor([b[0] for b in batch]), "input_ids": ids, "labels": lab, "attn_mask": mask}


def gpt_prompt_batch(src_list, pad=PAD_ID, bos=BOS_ID, sep=None):
    """Prompt sinh bản dịch: <s> nguồn <sep>, đệm TRÁI để mọi câu kết thúc cùng vị trí."""
    seqs = [[bos] + list(s) + [sep] for s in src_list]
    ids = _pad(seqs, pad, left=True)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    L = ids.size(1)
    for i, s in enumerate(seqs):
        mask[i, L - len(s):] = True
    return ids, mask


class TokenBatchSampler:
    """Gom câu có độ dài gần nhau vào cùng batch, giới hạn tổng số token (max_tokens).

    Tiết kiệm bộ nhớ và thời gian so với batch cố định số câu.  Thứ tự batch được xáo trộn
    theo seed + epoch => khi resume từ checkpoint, ta tái tạo ĐÚNG danh sách batch của epoch đó.
    """

    def __init__(self, lengths, max_tokens: int, shuffle: bool = True, seed: int = 0,
                 max_sentences: int | None = None):
        self.lengths = np.asarray(lengths)
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.max_sentences = max_sentences

    def batches(self, epoch: int = 0):
        rng = np.random.RandomState(self.seed + epoch)
        noise = rng.uniform(0, 1, len(self.lengths)) if self.shuffle else np.zeros(len(self.lengths))
        order = np.lexsort((noise, self.lengths))
        batches, cur, cur_max = [], [], 0
        for i in order:
            L = int(self.lengths[i])
            new_max = max(cur_max, L)
            too_many = self.max_sentences is not None and len(cur) + 1 > self.max_sentences
            if cur and (new_max * (len(cur) + 1) > self.max_tokens or too_many):
                batches.append(cur)
                cur, new_max = [], L
            cur.append(int(i))
            cur_max = new_max
        if cur:
            batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches


def sorted_eval_batches(lengths, batch_size: int):
    """Batch cho suy luận: sắp theo độ dài (nhanh hơn), trả về danh sách chỉ số gốc."""
    order = np.argsort(np.asarray(lengths), kind="stable")[::-1]
    return [order[i:i + batch_size].tolist() for i in range(0, len(order), batch_size)]
