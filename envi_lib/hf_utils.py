"""Làm việc với mô hình pretrained của Hugging Face — tương thích transformers 4.x và 5.x.

Những điểm khác biệt giữa các phiên bản đã được xử lý tập trung ở đây:
  * tham số kiểu số thực khi tải mô hình: `dtype` (mới) / `torch_dtype` (cũ)
  * `batch_decode` gộp vào `decode` ở v5
  * v5 mặc định use_cache=False => truyền use_cache=True khi generate để sinh nhanh
  * pipeline 'translation' bị bỏ ở v5 => không dùng pipeline
"""
from __future__ import annotations

import json
import os
import re

import torch

# DataLoader dùng tiến trình con (fork); tắt song song của tokenizers để tránh cảnh báo/treo
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def patch_peft_old_torchao():
    """Colab cài sẵn torchao bản cũ (0.10); peft >= 0.18 thấy torchao < 0.16 là báo ImportError ngay khi tạo LoRA,
    dù đồ án KHÔNG dùng torchao (chỉ dùng cho lượng tử hoá).  Ta báo cho peft là "không có torchao".
    Không gỡ/cài lại gói nào => không ảnh hưởng môi trường Colab."""
    import importlib
    import importlib.util
    import sys
    if importlib.util.find_spec("torchao") is None or importlib.util.find_spec("peft") is None:
        return False
    try:
        from importlib.metadata import version
        from packaging.version import Version
        if Version(version("torchao")) >= Version("0.16.0"):
            return False
    except Exception:  # noqa: BLE001
        return False
    patched = False
    for name in ("peft.import_utils", "peft.tuners.lora.torchao", "peft.utils.quantization_utils"):
        try:
            mod = sys.modules.get(name) or importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
        if hasattr(mod, "is_torchao_available"):
            mod.is_torchao_available = lambda: False
            patched = True
    if patched:
        print("ℹ️  Phát hiện torchao bản cũ của Colab — đã báo peft bỏ qua torchao (không dùng tới trong đồ án).")
    return patched


patch_peft_old_torchao()

from .common import autocast_ctx, normalize_text
from .data import sorted_eval_batches


def tf_version():
    import transformers
    from packaging.version import Version
    return Version(transformers.__version__)


def dtype_kw(dtype):
    """Tham số kiểu số thực cho from_pretrained theo phiên bản transformers."""
    from packaging.version import Version
    return {"dtype": dtype} if tf_version() >= Version("4.56.0") else {"torch_dtype": dtype}


def resolve_model_id(model_id: str) -> str:
    """Cho phép thay đường dẫn mô hình (dùng khi kiểm thử offline): ENVI_MODEL_OVERRIDES='{"id": "path"}'."""
    ov = os.environ.get("ENVI_MODEL_OVERRIDES")
    if ov:
        return json.loads(ov).get(model_id, model_id)
    return model_id


def decode_batch(tok, ids):
    if torch.is_tensor(ids):
        ids = ids.tolist()
    if hasattr(tok, "batch_decode"):
        return tok.batch_decode(ids, skip_special_tokens=True)
    out = tok.decode(ids, skip_special_tokens=True)
    return out if isinstance(out, list) else [out]


# ----------------------------------------------------------------------------
# Mô hình Encoder-Decoder pretrained (Marian, T5/envit5, mT5, NLLB)
# ----------------------------------------------------------------------------

class Seq2SeqSpec:
    """Mô tả cách dùng từng họ mô hình: tiền tố đầu vào, ngôn ngữ đích, hậu xử lý đầu ra."""

    def __init__(self, family: str, src_prefix: str = "", tgt_prefix: str = "",
                 src_lang: str | None = None, tgt_lang: str | None = None, fp16_ok: bool = True):
        self.family = family
        self.src_prefix = src_prefix
        self.tgt_prefix = tgt_prefix          # tiền tố trong NHÃN khi fine-tune (envit5: "vi: ")
        self.src_lang = src_lang              # NLLB: eng_Latn
        self.tgt_lang = tgt_lang              # NLLB: vie_Latn
        self.fp16_ok = fp16_ok                # họ T5 bị tràn số (NaN) với fp16

    def make_src(self, s):
        return self.src_prefix + s

    def make_tgt(self, t):
        return self.tgt_prefix + t

    def postprocess(self, out: str) -> str:
        out = normalize_text(out)
        if self.tgt_prefix and out.lower().startswith(self.tgt_prefix.strip().lower()):
            out = out[len(self.tgt_prefix.strip()):].lstrip(" :")
        return normalize_text(out)


# File cần để dựng lại tokenizer từ mô hình SentencePiece gốc (bỏ qua tokenizer.json)
_TOK_FILES = ["tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "config.json",
              "*.model", "*.spm", "vocab*.json", "vocab*.txt", "merges*.txt"]


def _tokenizer_from_sentencepiece(path, **tok_kw):
    """Dựng tokenizer KHÔNG dùng tokenizer.json, chỉ dùng file SentencePiece gốc (spiece.model ...).

    Dùng khi tokenizer.json của mô hình được tạo bằng phiên bản cũ, không đọc được bằng transformers 5.x
    (vd. VietAI/envit5-translation: lỗi "'dict' object is not an instance of 'Sequence'").
    """
    import shutil
    import tempfile
    from pathlib import Path
    from transformers import AutoTokenizer
    if os.path.isdir(path):
        src = Path(path)
    else:
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(path, allow_patterns=_TOK_FILES))
    if not any(src.glob("*.model")) and not any(src.glob("*.spm")):
        raise FileNotFoundError(f"{path}: không có file SentencePiece (*.model) để dựng lại tokenizer")
    try:
        import google.protobuf  # noqa: F401  (transformers cần protobuf để đọc file .model)
    except ImportError as e:
        raise ImportError("Cần gói protobuf để đọc spiece.model: chạy `%pip install protobuf` rồi "
                          "Runtime → Restart session, sau đó chạy lại notebook.") from e
    dst = Path(tempfile.mkdtemp(prefix="tok_"))
    for f in src.iterdir():
        if f.is_file() and f.name != "tokenizer.json":
            shutil.copy(f, dst / f.name)
    return AutoTokenizer.from_pretrained(str(dst), **tok_kw)


def load_tokenizer(path, **tok_kw):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(path, **tok_kw)
    except (TypeError, ValueError, KeyError, AttributeError, OSError) as e:
        if isinstance(e, OSError) and "tokenizer" not in str(e).lower():
            raise   # lỗi mạng / không tìm thấy mô hình: báo ngay, không che giấu
        print(f"⚠️  Không đọc được tokenizer.json của {path} ({type(e).__name__}: {str(e)[:120]}).\n"
              "   → Dựng lại tokenizer từ file SentencePiece gốc.")
        return _tokenizer_from_sentencepiece(path, **tok_kw)


def check_tokenizer(tok, model,
                    sample="Tôi là giáo viên Tin học ở Thành phố Hồ Chí Minh. I teach C++ to students."):
    """Kiểm tra tokenizer có khớp mô hình không.  Trả về (lỗi_nặng, số_token_<unk>_trên_câu_mẫu).

    Lỗi nặng: id <pad>/</s> khác mô hình; số token vượt xa bảng embedding.
    """
    hard = []
    V = model.get_input_embeddings().weight.shape[0]
    if len(tok) > 1.01 * V:
        hard.append(f"tokenizer có {len(tok)} token nhưng bảng embedding chỉ có {V}")
    cfg = model.config
    for name in ("pad_token_id", "eos_token_id"):
        a, b = getattr(tok, name, None), getattr(cfg, name, None)
        if a is not None and b is not None and not isinstance(b, list) and a != b:
            hard.append(f"{name}: tokenizer = {a}, mô hình = {b}")
    ids = tok(sample, add_special_tokens=False)["input_ids"]
    n_unk = ids.count(tok.unk_token_id) if tok.unk_token_id is not None else 0
    if len(ids) < len(sample.split()) // 2:
        hard.append(f"câu mẫu {len(sample.split())} từ chỉ được tách thành {len(ids)} token")
    return hard, n_unk


def load_seq2seq(model_id, spec: Seq2SeqSpec, device, for_training=False):
    from transformers import AutoModelForSeq2SeqLM
    path = resolve_model_id(model_id)
    tok_kw = {"src_lang": spec.src_lang, "tgt_lang": spec.tgt_lang} if spec.src_lang else {}
    tok = load_tokenizer(path, **tok_kw)
    model = AutoModelForSeq2SeqLM.from_pretrained(path, **dtype_kw(torch.float32)).to(device)
    hard, n_unk = check_tokenizer(tok, model)
    if hard or n_unk > 2:
        # tokenizer.json đọc được nhưng có dấu hiệu KHÔNG khớp mô hình => thử dựng lại từ SentencePiece gốc
        print(f"⚠️  Tokenizer nghi không khớp mô hình ({'; '.join(hard) or f'{n_unk} token <unk> trên câu mẫu'}).")
        try:
            tok2 = _tokenizer_from_sentencepiece(path, **tok_kw)
            hard2, n_unk2 = check_tokenizer(tok2, model)
            if not hard2 and (hard or n_unk2 < n_unk):
                print("   → Dùng tokenizer dựng lại từ file SentencePiece gốc.")
                tok, hard, n_unk = tok2, hard2, n_unk2
        except (FileNotFoundError, OSError, ValueError, TypeError) as e:
            print(f"   (không dựng lại được từ SentencePiece: {str(e)[:100]})")
    if hard:
        raise RuntimeError("Tokenizer không khớp mô hình: " + "; ".join(hard))
    if n_unk > 2:
        print(f"⚠️  Tokenizer sinh {n_unk} token <unk> trên câu mẫu — kết quả dịch có thể bị ảnh hưởng.")
    preflight_seq2seq(tok, model, spec)
    model.train(for_training)
    return tok, model


def preflight_seq2seq(tok, model, spec: Seq2SeqSpec):
    """Kiểm tra các yêu cầu đặc thù TRƯỚC khi chạy lâu."""
    vocab = tok.get_vocab()
    if spec.family == "marian":
        m = re.match(r"\s*(>>\w+<<)", spec.src_prefix)
        if m and m.group(1) not in vocab:
            raise RuntimeError(f"Tokenizer Marian không có token {m.group(1)} — kiểm tra lại mã ngôn ngữ đích.")
    if spec.tgt_lang:
        tid = tok.convert_tokens_to_ids(spec.tgt_lang)
        if tid is None or tid == tok.unk_token_id:
            raise RuntimeError(f"Không tìm thấy mã ngôn ngữ {spec.tgt_lang} trong tokenizer.")
    if model.config.decoder_start_token_id is None and getattr(model.generation_config, "decoder_start_token_id", None) is None:
        raise RuntimeError("Mô hình thiếu decoder_start_token_id.")


def seq2seq_gen_kwargs(tok, spec: Seq2SeqSpec, beam_size, max_new_tokens):
    # Encoder-decoder: max_length tính theo số token phía decoder (+1 token khởi đầu). Dùng max_length thay vì
    # max_new_tokens để ghi đè hẳn giá trị mặc định trong generation_config (Marian đặt sẵn 512) mà không bị cảnh báo.
    kw = dict(num_beams=beam_size, max_length=max_new_tokens + 1, do_sample=False, use_cache=True,
              length_penalty=1.0, early_stopping=beam_size > 1)
    if beam_size == 1:
        kw.pop("early_stopping")
    if spec.tgt_lang:
        kw["forced_bos_token_id"] = tok.convert_tokens_to_ids(spec.tgt_lang)
    return kw


@torch.no_grad()
def hf_seq2seq_translate(model, tok, spec: Seq2SeqSpec, srcs, beam_size=4, batch_size=32,
                         max_src_tokens=256, amp_dtype=None, progress=True):
    from tqdm.auto import tqdm
    model.eval()
    device = next(model.parameters()).device
    texts = [spec.make_src(s) for s in srcs]
    lens = [len(t) for t in texts]
    bs = max(1, batch_size // max(1, beam_size // 2))
    batches = sorted_eval_batches(lens, bs)
    out = [None] * len(texts)
    use_amp = amp_dtype if (amp_dtype != torch.float16 or spec.fp16_ok) else None
    it = tqdm(batches, desc=f"Dịch (beam={beam_size})", leave=False) if progress else batches
    for idx in it:
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_src_tokens)
        enc = {k: enc[k].to(device) for k in ("input_ids", "attention_mask")}  # bỏ token_type_ids nếu có
        max_new = int(enc["input_ids"].size(1) * 1.6 + 10)
        with autocast_ctx(use_amp):
            gen = model.generate(**enc, **seq2seq_gen_kwargs(tok, spec, beam_size, max_new))
        for i, h in zip(idx, decode_batch(tok, gen)):
            out[i] = spec.postprocess(h)
    return out


def make_hf_seq2seq_features(tok, spec: Seq2SeqSpec, srcs, tgts, max_len=256):
    """Token hoá cặp câu cho fine-tune.  Trả về list (input_ids, labels) và số cặp bị cắt."""
    enc = tok([spec.make_src(s) for s in srcs], text_target=[spec.make_tgt(t) for t in tgts],
              truncation=True, max_length=max_len)
    feats = list(zip(enc["input_ids"], enc["labels"]))
    n_trunc = sum(1 for a, b in feats if len(a) >= max_len or len(b) >= max_len)
    return feats, n_trunc


def hf_seq2seq_collate(batch, pad_id):
    """batch: list (idx, input_ids, labels)."""
    from .data import IGNORE, _pad
    ids = _pad([b[1] for b in batch], pad_id)
    lab = _pad([b[2] for b in batch], IGNORE)
    return {"input_ids": ids, "attention_mask": (ids != pad_id) if pad_id is not None else torch.ones_like(ids, dtype=torch.bool),
            "labels": lab}


def hf_seq2seq_loss(model, batch, cfg):
    """Tự tạo decoder_input_ids từ nhãn và tự tính loss (có label smoothing).

    Không truyền `labels` vào mô hình để tránh HF tính thêm một lần cross-entropy trên toàn bộ
    từ vựng (mT5/NLLB có 250k từ => tốn thêm vài GB bộ nhớ GPU).
    """
    from .models import label_smoothed_ce
    labels = batch["labels"]
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "prepare_decoder_input_ids_from_labels"):
        dec_in = base.prepare_decoder_input_ids_from_labels(labels=labels)
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"].long(),
                    decoder_input_ids=dec_in)
    else:
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"].long(), labels=labels)
    return label_smoothed_ce(out.logits, labels, cfg.label_smoothing)


class FeatDataset(torch.utils.data.Dataset):
    def __init__(self, feats):
        self.feats = feats

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        a, b = self.feats[i]
        return i, a, b


# ----------------------------------------------------------------------------
# LLM decoder-only (Qwen ...) — prompt, LoRA
# ----------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a professional English-Vietnamese translator."
USER_TEMPLATE = ("Translate the following English sentence into Vietnamese. "
                 "Output only the Vietnamese translation, nothing else.\n\n{src}")


def build_messages(src, shots=()):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s, t in shots:
        msgs.append({"role": "user", "content": USER_TEMPLATE.format(src=s)})
        msgs.append({"role": "assistant", "content": t})
    msgs.append({"role": "user", "content": USER_TEMPLATE.format(src=src)})
    return msgs


def chat_prompt(tok, src, shots=()):
    kw = {}
    if "enable_thinking" in (getattr(tok, "chat_template", None) or ""):
        kw["enable_thinking"] = False   # Qwen3: tắt chế độ "suy nghĩ"
    s = tok.apply_chat_template(build_messages(src, shots), tokenize=False, add_generation_prompt=True, **kw)
    if not isinstance(s, str):
        raise RuntimeError("apply_chat_template(tokenize=False) không trả về chuỗi.")
    return s


def chat_train_texts(tok, src, tgt):
    """(prompt_text, target_text) sao cho prompt + target = hội thoại đầy đủ theo chat template."""
    prompt = chat_prompt(tok, src)
    full = tok.apply_chat_template(build_messages(src) + [{"role": "assistant", "content": tgt}],
                                   tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise RuntimeError("Chat template không nhất quán giữa prompt và hội thoại đầy đủ "
                           "(thường gặp ở mô hình có chế độ suy nghĩ). Hãy dùng Qwen2.5-Instruct.")
    target = full[len(prompt):]
    eos = getattr(tok, "eos_token", None)
    if eos and eos in target:   # bỏ phần sau token kết thúc (vd. "\n" của ChatML) — không cần học
        target = target[: target.index(eos) + len(eos)]
    return prompt, target


_PREFIX_RE = re.compile(r"^(vietnamese( translation)?|translation|bản dịch|dịch|tiếng việt)\s*[:：]\s*", re.I)


def clean_llm_output(text: str) -> str:
    """Hậu xử lý đầu ra LLM: lấy dòng đầu không rỗng, bỏ tiền tố 'Translation:' và dấu ngoặc kép bao ngoài."""
    text = (text or "").replace("<think>", "").replace("</think>", "")
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    s = lines[0] if lines else ""
    s = _PREFIX_RE.sub("", s)
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'“”":
        s = s[1:-1]
    if len(s) >= 2 and s[0] == "“" and s[-1] == "”":
        s = s[1:-1]
    return normalize_text(s)


def load_llm(model_id, device, dtype):
    from transformers import AutoModelForCausalLM
    path = resolve_model_id(model_id)
    tok = load_tokenizer(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if not getattr(tok, "chat_template", None):
        raise RuntimeError(f"{model_id} không có chat template — hãy dùng bản *-Instruct.")
    model = AutoModelForCausalLM.from_pretrained(path, **dtype_kw(dtype)).to(device)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tok.pad_token_id
    return tok, model


def make_llm_features(tok, srcs, tgts, max_len=384):
    feats, n_skip = [], 0
    for s, t in zip(srcs, tgts):
        p, tgt = chat_train_texts(tok, s, t)
        p_ids = tok(p, add_special_tokens=False)["input_ids"]
        t_ids = tok(tgt, add_special_tokens=False)["input_ids"]
        if len(p_ids) + len(t_ids) > max_len:
            n_skip += 1
            continue
        feats.append((p_ids, t_ids))
    return feats, n_skip


def llm_collate(batch, pad_id):
    from .data import IGNORE
    seqs = [list(p) + list(t) for _, p, t in batch]
    labs = [[IGNORE] * len(p) + list(t) for _, p, t in batch]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    lab = torch.full((len(seqs), L), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        ids[i, :len(s)] = torch.tensor(s)
        lab[i, :len(l)] = torch.tensor(l)
        mask[i, :len(s)] = 1
    return {"input_ids": ids, "labels": lab, "attention_mask": mask}


def llm_loss(model, batch, cfg):
    from .models import label_smoothed_ce
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1]
    labels = batch["labels"][:, 1:]
    return label_smoothed_ce(logits, labels, cfg.label_smoothing)


@torch.no_grad()
def llm_translate(model, tok, srcs, shots=(), batch_size=32, max_new_tokens=256, amp_dtype=None,
                  progress=True, return_raw=False):
    from tqdm.auto import tqdm
    model.eval()
    device = next(model.parameters()).device
    old_side = tok.padding_side
    tok.padding_side = "left"   # sinh văn bản theo batch với decoder-only: BẮT BUỘC đệm trái
    prompts = [chat_prompt(tok, s, shots) for s in srcs]
    batches = sorted_eval_batches([len(p) for p in prompts], batch_size)
    out, raw = [None] * len(srcs), [None] * len(srcs)
    eos = model.generation_config.eos_token_id
    it = tqdm(batches, desc="LLM dịch", leave=False) if progress else batches
    try:
        for idx in it:
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: enc[k].to(device) for k in ("input_ids", "attention_mask")}
            with autocast_ctx(amp_dtype):
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                     use_cache=True, pad_token_id=tok.pad_token_id, eos_token_id=eos)
            new = gen[:, enc["input_ids"].size(1):]
            for i, h in zip(idx, decode_batch(tok, new)):
                raw[i] = h
                out[i] = clean_llm_output(h)
    finally:
        tok.padding_side = old_side
    return (out, raw) if return_raw else out


def enable_grad_ckpt(peft_model):
    """Gradient checkpointing: giảm bộ nhớ kích hoạt (activation) ~3-5 lần, đổi lại chậm hơn ~30%."""
    base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    try:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        base.gradient_checkpointing_enable()
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    base.config.use_cache = False
