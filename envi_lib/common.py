"""Tiện ích dùng chung cho toàn bộ đồ án dịch máy Anh -> Việt.

Mọi notebook đều import module này để dùng chung: đường dẫn, seed, chuẩn hoá văn bản,
chọn kiểu số thực (AMP), lưu/đọc file dự đoán.  Giữ một nguồn duy nhất để tránh mỗi
notebook xử lý một kiểu dẫn đến số liệu không so sánh được.
"""
from __future__ import annotations

import contextlib
import datetime
import json
import os
import random
import re
import shutil
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch

LIB_VERSION = "1.0.0"

# ----------------------------------------------------------------------------
# Môi trường & đường dẫn
# ----------------------------------------------------------------------------

def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


DEFAULT_DRIVE_DIR = "/content/drive/MyDrive/NLP_Final_EnVi"


class Paths:
    """Cấu trúc thư mục của đồ án.

    root/
      data_raw/       dữ liệu gốc tải về (dùng chung cho DRY_RUN và chạy thật)
      data_clean/     dữ liệu đã làm sạch (dùng chung)
      spm/            tokenizer SentencePiece
      checkpoints/    checkpoint huấn luyện
      predictions/    bản dịch tập test của từng hệ thống (*.vi.txt + *.meta.json)
      results/        bảng số liệu, hình vẽ
    Khi dry_run=True, các thư mục spm/checkpoints/predictions/results nằm trong root/dryrun/
    để kết quả chạy thử không lẫn vào kết quả thật.
    """

    def __init__(self, root: str | os.PathLike, dry_run: bool = False):
        self.root = Path(root)
        self.dry_run = dry_run
        work = self.root / "dryrun" if dry_run else self.root
        self.work = work
        self.data_raw = self.root / "data_raw"
        self.data_clean = self.root / "data_clean"
        self.spm = work / "spm"
        self.checkpoints = work / "checkpoints"
        self.predictions = work / "predictions"
        self.results = work / "results"
        self.figures = self.results / "figures"

    def make_all(self):
        for p in [self.data_raw, self.data_clean, self.spm, self.checkpoints,
                  self.predictions, self.results, self.figures]:
            p.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def spm_model(self) -> Path:
        return self.spm / "spm_envi.model"

    def __repr__(self):
        return f"Paths(root={self.root}, dry_run={self.dry_run})"


def get_project_root(default: str = DEFAULT_DRIVE_DIR) -> Path:
    """Thư mục gốc của đồ án.  Có thể ghi đè bằng biến môi trường ENVI_PROJECT_DIR."""
    env = os.environ.get("ENVI_PROJECT_DIR")
    return Path(env) if env else Path(default)


# ----------------------------------------------------------------------------
# Seed, thiết bị, AMP
# ----------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bf16_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def pick_amp_dtype(prefer: str = "auto"):
    """Chọn kiểu số thực cho autocast.

    - GPU hỗ trợ bf16 (A100, L4, H100): bf16 — ổn định, không cần GradScaler.
    - GPU cũ (T4, V100): fp16 + GradScaler.
    - CPU: None (chạy fp32).
    prefer: "auto" | "bf16" | "fp16" | "fp32"
    """
    if not torch.cuda.is_available() or prefer == "fp32":
        return None
    if prefer == "fp16":
        return torch.float16
    if prefer == "bf16":
        if not bf16_supported():
            raise RuntimeError("GPU này không hỗ trợ bf16. Dùng prefer='auto' hoặc 'fp16'.")
        return torch.bfloat16
    return torch.bfloat16 if bf16_supported() else torch.float16


def autocast_ctx(amp_dtype):
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def gpu_info() -> dict:
    info = {"cuda": torch.cuda.is_available(), "torch": torch.__version__}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(name=p.name, total_gb=round(p.total_memory / 1024**3, 1),
                    bf16=torch.cuda.is_bf16_supported())
    return info


def peak_gpu_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return round(torch.cuda.max_memory_allocated() / 1024**3, 2)


def reset_peak_mem():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def free_memory():
    """Giải phóng bộ nhớ GPU giữa các mô hình.  Gọi SAU khi đã `del model, optimizer, ...`."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def count_params(model, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def disk_free_gb(path) -> float:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return round(shutil.disk_usage(path).free / 1024**3, 1)


# ----------------------------------------------------------------------------
# Chuẩn hoá văn bản
# ----------------------------------------------------------------------------

_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")
_SPACES = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Chuẩn hoá nhẹ, áp dụng thống nhất cho nguồn, tham chiếu và bản dịch.

    - Unicode NFC: tiếng Việt có thể ở dạng dựng sẵn hoặc tổ hợp (ví dụ 'ệ' = 1 hay 3 ký tự).
      Hai dạng nhìn giống nhau nhưng BLEU coi là khác nhau => bắt buộc chuẩn hoá.
    - Bỏ ký tự độ rộng 0, gộp khoảng trắng (kể cả xuống dòng) thành 1 dấu cách.
    Không đổi chữ hoa/thường, không đổi dấu câu (để không làm sai lệch tham chiếu).
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = _ZERO_WIDTH.sub("", s)
    s = _SPACES.sub(" ", s)
    return s.strip()


# ----------------------------------------------------------------------------
# Đọc / ghi file
# ----------------------------------------------------------------------------

def read_lines(path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").rstrip("\r") for line in f]


def write_lines(path, lines):
    """Ghi mỗi phần tử một dòng.  Ghi ra file tạm rồi đổi tên (an toàn khi bị ngắt giữa chừng)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for line in lines:
            line = "" if line is None else str(line)
            if "\n" in line or "\r" in line:
                line = _SPACES.sub(" ", line)
            f.write(line + "\n")
    os.replace(tmp, path)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = self.elapsed()


def fmt_seconds(s: float) -> str:
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# ----------------------------------------------------------------------------
# Lưu bản dịch của từng hệ thống
# ----------------------------------------------------------------------------

SYSTEM_NAME_RE = re.compile(r"^[A-Za-z0-9_\-\.]+$")


def save_predictions(paths: Paths, system: str, hyps: list[str], meta: dict | None = None,
                     expected_n: int | None = None):
    """Lưu bản dịch tập test: predictions/<system>.vi.txt và <system>.meta.json.

    Kiểm tra: số dòng phải bằng số câu test; cảnh báo nếu có nhiều câu rỗng.
    """
    if not SYSTEM_NAME_RE.match(system):
        raise ValueError(f"Tên hệ thống '{system}' chỉ được chứa chữ, số, '_', '-', '.'")
    hyps = [normalize_text(h) for h in hyps]
    if expected_n is not None and len(hyps) != expected_n:
        raise ValueError(f"[{system}] Có {len(hyps)} bản dịch nhưng tập test có {expected_n} câu.")
    n_empty = sum(1 for h in hyps if not h)
    if n_empty:
        print(f"⚠️  [{system}] {n_empty} bản dịch rỗng ({n_empty / max(1, len(hyps)):.1%}).")
    paths.predictions.mkdir(parents=True, exist_ok=True)
    write_lines(paths.predictions / f"{system}.vi.txt", hyps)
    meta = dict(meta or {})
    meta.update(system=system, n_lines=len(hyps), n_empty=n_empty, saved_at=now_str(),
                lib_version=LIB_VERSION, gpu=gpu_info().get("name", "cpu"))
    save_json(meta, paths.predictions / f"{system}.meta.json")
    print(f"✅ Đã lưu {len(hyps)} bản dịch: {paths.predictions / (system + '.vi.txt')}")


def check_lib_version(required: str):
    if LIB_VERSION != required:
        raise RuntimeError(
            f"Thư viện envi_lib trên Drive là bản {LIB_VERSION}, notebook này cần bản {required}. "
            "Hãy chạy lại notebook 00_cai_dat.ipynb để cập nhật thư viện.")


def prediction_is_current(paths: Paths, system: str, fingerprint: dict, expected_n: int) -> bool:
    """True nếu bản dịch của hệ thống đã tồn tại và được sinh từ đúng checkpoint/cấu hình hiện tại.

    Dùng để KHÔNG dịch lại tập test một cách thừa khi chạy lại notebook (tiết kiệm thời gian GPU).
    """
    mp = paths.predictions / f"{system}.meta.json"
    tp = paths.predictions / f"{system}.vi.txt"
    if not (mp.exists() and tp.exists()):
        return False
    try:
        meta = load_json(mp)
    except Exception:  # noqa: BLE001
        return False
    if meta.get("n_lines") != expected_n or meta.get("fingerprint") != fingerprint:
        return False
    print(f"✔ {system}: đã có bản dịch từ đúng checkpoint này — bỏ qua (xoá file trong predictions/ nếu muốn dịch lại).")
    return True
