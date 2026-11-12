"""Tạo môi trường Python RIÊNG cho COMET + BERTScore và chạy eval_neural.py trong đó.

Vì sao: `unbabel-comet` đòi numpy<2, transformers<5, huggingface-hub<1 — cài thẳng vào Colab sẽ
hạ cấp các thư viện đang dùng và làm hỏng notebook.  Môi trường riêng (venv) giải quyết triệt để:
hai bộ thư viện không đụng nhau.

PyTorch trong venv được cài CÙNG phiên bản + CÙNG bản CUDA với PyTorch của Colab
=> chắc chắn tương thích driver GPU của máy.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# setuptools<81: torchmetrics 0.10 (phụ thuộc của COMET) còn dùng pkg_resources, đã bị gỡ khỏi setuptools mới
EVAL_PKGS = ["unbabel-comet==2.2.7", "bert-score==0.3.13", "setuptools<81"]


def _run(cmd, log=print):
    log("$ " + " ".join(map(str, cmd)))
    p = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-40:]
        if any(k in line for k in ("COMET", "BERTScore", "XONG", "!!", "Error", "error:", "Traceback")):
            log("   " + line)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"Lệnh lỗi (mã {p.returncode}):\n" + "\n".join(tail))


def ensure_eval_env(env_dir="/content/eval_env", python: str | None = None, log=print) -> Path:
    env_dir = Path(env_dir)
    py = env_dir / "bin" / "python"
    marker = env_dir / ".ready"
    if marker.exists() and py.exists():
        log(f"✔ Môi trường đánh giá đã sẵn sàng: {env_dir}")
        return py
    import torch
    _run([sys.executable, "-m", "pip", "install", "-q", "uv"], log)
    _run([sys.executable, "-m", "uv", "venv", "--clear", env_dir, "--python", python or sys.executable], log)
    base, _, variant = torch.__version__.partition("+")
    installed = False
    if variant.startswith("cu") or variant == "cpu":
        try:
            _run([sys.executable, "-m", "uv", "pip", "install", "--python", py, f"torch=={base}",
                  "--index-url", f"https://download.pytorch.org/whl/{variant}"], log)
            installed = True
        except RuntimeError as e:
            log(f"⚠️  Không cài được torch từ download.pytorch.org ({str(e)[-200:]}); thử PyPI.")
    if not installed:
        _run([sys.executable, "-m", "uv", "pip", "install", "--python", py, f"torch=={base}"], log)
    _run([sys.executable, "-m", "uv", "pip", "install", "--python", py, *EVAL_PKGS], log)
    _run([py, "-c", "import comet, bert_score, torch, transformers, numpy; "
                    "print('OK torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
                    "'transformers', transformers.__version__, 'numpy', numpy.__version__)"], log)
    marker.write_text("ok")
    log(f"✔ Đã tạo môi trường đánh giá: {env_dir}")
    return py


def run_neural_metrics(py, script, src_file, ref_file, hyp_files, out_json, comet=True, bertscore=True,
                       batch=64, bertscore_model="xlm-roberta-large", extra_args=(), log=print):
    cmd = [py, script, "--src", src_file, "--ref", ref_file, "--out", out_json, "--batch", str(batch),
           "--bertscore-model", bertscore_model, "--hyp", *hyp_files, *extra_args]
    if not comet:
        cmd.append("--no-comet")
    if not bertscore:
        cmd.append("--no-bertscore")
    _run(cmd, log)
