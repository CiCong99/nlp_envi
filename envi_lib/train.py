"""Vòng lặp huấn luyện dùng chung cho MỌI mô hình (tự cài, HF seq2seq, LLM + LoRA).

Các cơ chế phòng rủi ro:
  * Checkpoint 'last.pt' lưu định kỳ (model + optimizer + scheduler + scaler + RNG + vị trí batch).
    Colab mất kết nối => chạy lại ô huấn luyện là tự động tiếp tục ĐÚNG chỗ cũ.
  * Ghi file tạm rồi đổi tên => không bao giờ có checkpoint hỏng nửa chừng.
  * Lưu 'best' theo điểm dev (BLEU) + early stopping.
  * Phát hiện loss NaN/inf: bỏ qua bước đó; quá nhiều lần liên tiếp => dừng và báo nguyên nhân.
  * Bắt lỗi hết bộ nhớ GPU (OOM): bỏ qua batch; lặp lại nhiều lần => dừng và gợi ý giảm max_tokens.
  * Giới hạn thời gian (time_budget_hours): dừng an toàn trước khi hết quota Colab.
  * Ngắt tay (Interrupt): vẫn lưu checkpoint trước khi dừng.
"""
from __future__ import annotations

import csv
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .common import LIB_VERSION, autocast_ctx, fmt_seconds, now_str, save_json


@dataclass
class TrainConfig:
    epochs: int = 50
    max_tokens: int = 8192            # số token tối đa / batch (trước tích luỹ gradient)
    max_sentences: int | None = None  # giới hạn số câu / batch (tuỳ chọn)
    grad_accum: int = 1
    lr: float = 7e-4
    warmup_steps: int = 4000
    schedule: str = "inverse_sqrt"    # inverse_sqrt | linear | cosine | constant
    min_lr_ratio: float = 0.0         # cho linear/cosine
    weight_decay: float = 0.0001
    betas: tuple = (0.9, 0.98)
    adam_eps: float = 1e-8
    clip_norm: float = 1.0
    label_smoothing: float = 0.1
    patience: int = 6                 # số lần đánh giá liên tiếp không cải thiện thì dừng
    eval_every_steps: int | None = None  # None = đánh giá cuối mỗi epoch
    save_every_steps: int = 1000      # lưu last.pt định kỳ giữa epoch
    log_every_steps: int = 200
    max_steps: int | None = None
    time_budget_hours: float | None = None
    seed: int = 42
    amp: str = "auto"                 # auto | bf16 | fp16 | fp32
    save_optimizer: bool = True

    def to_dict(self):
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d


def make_scheduler(optimizer, cfg: TrainConfig, total_steps: int):
    w = max(1, cfg.warmup_steps)

    def f(step):
        step = max(1, step)
        if cfg.schedule == "inverse_sqrt":
            return min(step / w, math.sqrt(w / step))
        if step < w:
            return step / w
        if cfg.schedule == "constant":
            return 1.0
        prog = min(1.0, (step - w) / max(1, total_steps - w))
        if cfg.schedule == "linear":
            return max(cfg.min_lr_ratio, 1 - prog * (1 - cfg.min_lr_ratio))
        if cfg.schedule == "cosine":
            return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
        raise ValueError(cfg.schedule)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def _atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state():
    st = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    try:
        random.setstate(st["py"])
        np.random.set_state(st["np"])
        torch.set_rng_state(st["torch"])
        if "cuda" in st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["cuda"])
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Không khôi phục được trạng thái ngẫu nhiên ({e}) — vẫn tiếp tục.")


class TooManyFailures(RuntimeError):
    pass


class Trainer:
    """
    loss_fn(model, batch, cfg) -> (loss, nll_mean, n_tokens)   [batch đã ở trên device]
    eval_fn(model) -> dict, bắt buộc có khoá 'score' (càng cao càng tốt, ví dụ BLEU dev)
    save_best_fn(model, dir) -> None : tuỳ chọn, cách lưu mô hình tốt nhất (mặc định state_dict)
    """

    def __init__(self, model, dataset, collate_fn, lengths, cfg: TrainConfig, ckpt_dir,
                 loss_fn: Callable, eval_fn: Callable | None = None, device=None,
                 save_best_fn: Callable | None = None, run_name: str = "run",
                 extra_meta: dict | None = None, params=None, last_dir=None):
        self.model = model
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.lengths = lengths
        self.cfg = cfg
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        # last.pt (kèm trạng thái optimizer, có thể nặng vài GB) có thể đặt ở ổ máy ảo thay vì Drive
        self.last_dir = Path(last_dir) if last_dir else self.ckpt_dir
        self.last_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.device = device or next(model.parameters()).device
        self.save_best_fn = save_best_fn
        self.run_name = run_name
        self.extra_meta = extra_meta or {}
        from .common import pick_amp_dtype
        self.amp_dtype = pick_amp_dtype(cfg.amp) if self.device.type == "cuda" else None
        self.use_scaler = self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        from .data import TokenBatchSampler
        self.sampler = TokenBatchSampler(lengths, cfg.max_tokens, shuffle=True, seed=cfg.seed,
                                         max_sentences=cfg.max_sentences)
        self.steps_per_epoch = math.ceil(len(self.sampler.batches(0)) / cfg.grad_accum)
        self.total_steps = cfg.max_steps or self.steps_per_epoch * cfg.epochs

        params = [p for p in (params if params is not None else model.parameters()) if p.requires_grad]
        decay = [p for p in params if p.ndim >= 2]
        no_decay = [p for p in params if p.ndim < 2]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=cfg.lr, betas=tuple(cfg.betas), eps=cfg.adam_eps)
        self.scheduler = make_scheduler(self.optimizer, cfg, self.total_steps)

        self.state = {"epoch": 0, "batch_in_epoch": 0, "global_step": 0, "best_score": -float("inf"),
                      "best_step": 0, "bad_evals": 0, "train_seconds": 0.0, "finished": False}
        self.history = []

    # ---------------------------------------------------------------- checkpoint
    @property
    def last_path(self):
        return self.last_dir / "last.pt"

    def _trainable_state(self):
        # Với LoRA chỉ lưu tham số được huấn luyện (vài chục MB thay vì vài GB)
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()
                if k in self._trainable_names}

    @property
    def _trainable_names(self):
        if not hasattr(self, "_tn"):
            names = {n for n, p in self.model.named_parameters() if p.requires_grad}
            all_params = {n for n, _ in self.model.named_parameters()}
            # Nếu train toàn bộ thì lưu cả state_dict (gồm buffer)
            self._tn = None if names == all_params else names
        return self._tn

    def save_last(self):
        sd = self.model.state_dict() if self._trainable_names is None else self._trainable_state()
        ck = {"model": sd, "state": self.state, "history": self.history, "rng": _rng_state(),
              "cfg": self.cfg.to_dict(), "lib_version": LIB_VERSION, "saved_at": now_str(),
              "scheduler": self.scheduler.state_dict(), "scaler": self.scaler.state_dict()}
        if self.cfg.save_optimizer:
            ck["optimizer"] = self.optimizer.state_dict()
        _atomic_save(ck, self.last_path)

    def try_resume(self) -> bool:
        if not self.last_path.exists():
            return False
        ck = torch.load(self.last_path, map_location="cpu", weights_only=False)
        old = ck.get("cfg", {})
        important = ["lr", "max_tokens", "grad_accum", "schedule", "warmup_steps", "seed", "epochs"]
        diffs = {k: (old.get(k), self.cfg.to_dict().get(k)) for k in important if old.get(k) != self.cfg.to_dict().get(k)}
        if diffs:
            print(f"⚠️  Cấu hình khác với checkpoint cũ: {diffs}. Tiếp tục với cấu hình MỚI "
                  "(nếu muốn huấn luyện lại từ đầu: xoá thư mục checkpoint của run này).")
        try:
            missing, unexpected = self.model.load_state_dict(ck["model"], strict=self._trainable_names is None)
        except RuntimeError as e:
            raise RuntimeError(
                f"Checkpoint {self.last_path} không khớp với mô hình hiện tại (thường do đã đổi mô hình/cấu hình "
                f"nhưng còn checkpoint cũ). Xoá file đó (và thư mục {self.ckpt_dir}) rồi chạy lại.\n{str(e)[:300]}") from None
        if self._trainable_names is not None and unexpected:
            raise RuntimeError(f"Checkpoint có tham số lạ: {unexpected[:5]}")
        if "optimizer" in ck:
            self.optimizer.load_state_dict(ck["optimizer"])
        else:
            print("ℹ️  Checkpoint không có trạng thái optimizer — khởi tạo lại optimizer.")
        self.scheduler.load_state_dict(ck["scheduler"])
        if self.use_scaler and ck.get("scaler"):
            self.scaler.load_state_dict(ck["scaler"])
        self.state.update(ck["state"])
        self.history = ck.get("history", [])
        _set_rng_state(ck.get("rng", {}))
        print(f"🔁 Tiếp tục từ checkpoint ({ck.get('saved_at')}): epoch {self.state['epoch'] + 1}, "
              f"batch {self.state['batch_in_epoch']}, step {self.state['global_step']}, "
              f"best dev = {self.state['best_score']:.2f}")
        return True

    def _save_best(self):
        if self.save_best_fn is not None:
            self.save_best_fn(self.model, self.ckpt_dir / "best")
        else:
            _atomic_save({"model": self.model.state_dict(), "step": self.state["global_step"],
                          "score": self.state["best_score"], "meta": self.extra_meta}, self.ckpt_dir / "best.pt")

    def _log_history(self, row):
        self.history.append(row)
        path = self.ckpt_dir / "history.csv"
        keys = sorted({k for r in self.history for k in r})
        tmp = path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.history)
        os.replace(tmp, path)

    # ---------------------------------------------------------------- evaluate
    def evaluate(self, tag=""):
        row = {"step": self.state["global_step"], "epoch": self.state["epoch"],
               "lr": self.optimizer.param_groups[0]["lr"], "time": now_str()}
        self.state["last_eval_step"] = self.state["global_step"]
        if self.eval_fn is not None:
            self.model.eval()
            res = self.eval_fn(self.model)
            self.model.train()
            row.update(res)
            score = res["score"]
            improved = score > self.state["best_score"] + 1e-6
            if improved:
                self.state["best_score"], self.state["best_step"], self.state["bad_evals"] = score, self.state["global_step"], 0
                self._save_best()
            else:
                self.state["bad_evals"] += 1
            row["best"] = improved
            extra = " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items() if k != "score")
            if improved:
                note = "⭐ tốt nhất mới"
            else:
                note = "(không cải thiện %d/%d)" % (self.state["bad_evals"], self.cfg.patience)
            print(f"   📊 {tag} step {self.state['global_step']}: dev score={score:.2f} {extra} {note}")
        self._log_history(row)

    # ---------------------------------------------------------------- train
    def train(self):
        from tqdm.auto import tqdm
        cfg = self.cfg
        self.try_resume()
        if self.state["finished"]:
            reason = self.state.get("stop_reason")
            if reason == "đủ số epoch" and self.state["epoch"] < cfg.epochs:
                print(f"➕ Tăng số epoch lên {cfg.epochs}: huấn luyện tiếp.")
                self.state["finished"] = False
            elif reason == "đạt max_steps" and (cfg.max_steps is None or self.state["global_step"] < cfg.max_steps):
                print("➕ Tăng/bỏ giới hạn max_steps: huấn luyện tiếp.")
                self.state["finished"] = False
        if self.state["finished"]:
            print("✔ Run này đã huấn luyện xong trước đó. (Xoá thư mục checkpoint của run nếu muốn huấn luyện lại.)")
            return self.history
        if cfg.max_steps and self.state["global_step"] >= cfg.max_steps:
            print(f"✔ Đã đạt max_steps={cfg.max_steps}.")
            return self.history
        self.model.train()
        t_start = time.time()
        sec_before = self.state["train_seconds"]
        n_bad_loss = n_oom = 0
        stop_reason = None
        loss_acc = nll_acc = tok_acc = 0.0
        since_log = 0

        try:
            for epoch in range(self.state["epoch"], cfg.epochs):
                self.state["epoch"] = epoch
                batches = self.sampler.batches(epoch)
                start_b = self.state["batch_in_epoch"]
                loader = torch.utils.data.DataLoader(self.dataset, batch_sampler=batches[start_b:],
                                                     collate_fn=self.collate_fn, num_workers=2,
                                                     persistent_workers=False, pin_memory=self.device.type == "cuda",
                                                     # generator riêng => DataLoader không "ăn" RNG toàn cục,
                                                     # dropout sau khi resume giống hệt khi chạy liền mạch
                                                     generator=torch.Generator().manual_seed(cfg.seed + epoch))
                pbar = tqdm(total=len(batches), initial=start_b, desc=f"{self.run_name} epoch {epoch + 1}/{cfg.epochs}",
                            leave=False)
                self.optimizer.zero_grad(set_to_none=True)
                micro = 0
                for bi, batch in enumerate(loader, start=start_b):
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    try:
                        with autocast_ctx(self.amp_dtype):
                            loss, nll, ntok = self.loss_fn(self.model, batch, cfg)
                        if not torch.isfinite(loss):
                            n_bad_loss += 1
                            self.optimizer.zero_grad(set_to_none=True)
                            micro = 0
                            if n_bad_loss >= 20:
                                raise TooManyFailures(
                                    "Loss bị NaN/inf 20 lần liên tiếp. Thử: giảm lr, đặt amp='fp32' hoặc 'bf16'.")
                            continue
                        n_bad_loss = 0
                        self.scaler.scale(loss / cfg.grad_accum).backward()
                    except torch.cuda.OutOfMemoryError:
                        n_oom += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        micro = 0
                        del batch
                        torch.cuda.empty_cache()
                        print(f"⚠️  Hết bộ nhớ GPU ở batch {bi} (lần {n_oom}) — bỏ qua batch này.")
                        if n_oom >= 10:
                            raise TooManyFailures("Hết bộ nhớ GPU quá nhiều lần. Giảm max_tokens (vd. chia đôi) "
                                                  "và tăng grad_accum tương ứng.")
                        continue
                    loss_acc += float(loss.detach()) * ntok
                    nll_acc += float(nll) * ntok
                    tok_acc += ntok
                    micro += 1
                    pbar.update(1)
                    self.state["batch_in_epoch"] = bi + 1
                    if micro < cfg.grad_accum:
                        continue
                    micro = 0
                    self.scaler.unscale_(self.optimizer)
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        [p for g in self.optimizer.param_groups for p in g["params"]], cfg.clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.state["global_step"] += 1
                    since_log += 1
                    gs = self.state["global_step"]

                    if gs % cfg.log_every_steps == 0 and tok_acc > 0:
                        tl = loss_acc / tok_acc
                        ppl = math.exp(min(20, nll_acc / tok_acc))
                        pbar.set_postfix(loss=f"{tl:.3f}", ppl=f"{ppl:.1f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
                        self._log_history({"step": gs, "epoch": epoch, "train_loss": tl, "train_ppl": ppl,
                                           "grad_norm": float(gnorm), "lr": self.optimizer.param_groups[0]["lr"],
                                           "time": now_str()})
                        loss_acc = nll_acc = tok_acc = 0.0
                    if cfg.eval_every_steps and gs % cfg.eval_every_steps == 0:
                        self.evaluate(tag=f"epoch {epoch + 1}")
                        if self.state["bad_evals"] >= cfg.patience:
                            stop_reason = "early stopping"
                    if gs % cfg.save_every_steps == 0:
                        self.state["train_seconds"] = sec_before + time.time() - t_start
                        self.save_last()
                    if cfg.max_steps and gs >= cfg.max_steps:
                        stop_reason = stop_reason or "đạt max_steps"
                    if cfg.time_budget_hours and (time.time() - t_start) / 3600 > cfg.time_budget_hours:
                        stop_reason = stop_reason or "hết thời gian cho phép (time_budget_hours)"
                    if stop_reason:
                        break
                pbar.close()
                if stop_reason:
                    break
                # cuối epoch
                self.state["epoch"] = epoch + 1
                self.state["batch_in_epoch"] = 0
                if not cfg.eval_every_steps:
                    self.evaluate(tag=f"epoch {epoch + 1}/{cfg.epochs}")
                    if self.state["bad_evals"] >= cfg.patience:
                        stop_reason = "early stopping"
                self.state["train_seconds"] = sec_before + time.time() - t_start
                self.save_last()
                if stop_reason:
                    break
        except KeyboardInterrupt:
            self.state["train_seconds"] = sec_before + time.time() - t_start
            self.save_last()
            print("⏸ Đã dừng tay và lưu checkpoint. Chạy lại ô này để tiếp tục.")
            raise

        self.state["train_seconds"] = sec_before + time.time() - t_start
        # Chỉ coi là "xong hẳn" khi dừng sớm hoặc đủ epoch; dừng vì max_steps/hết giờ thì lần sau chạy tiếp được
        # Đánh giá lần cuối nếu bước hiện tại chưa được đánh giá (vd. dừng vì max_steps giữa epoch)
        if (self.eval_fn is not None and stop_reason != "hết thời gian cho phép (time_budget_hours)"
                and self.state.get("last_eval_step") != self.state["global_step"]):
            self.evaluate(tag="cuối")
        if stop_reason in (None, "early stopping", "đạt max_steps"):
            self.state["finished"] = True
        self.state["stop_reason"] = stop_reason or "đủ số epoch"
        self.save_last()
        save_json({"run": self.run_name, **self.state,
                   "train_time": fmt_seconds(self.state["train_seconds"]), "cfg": self.cfg.to_dict(),
                   **self.extra_meta}, self.ckpt_dir / "train_summary.json")
        print(f"🏁 {self.run_name}: kết thúc ({stop_reason or 'đủ số epoch'}), "
              f"best dev = {self.state['best_score']:.2f} tại step {self.state['best_step']}, "
              f"thời gian train = {fmt_seconds(self.state['train_seconds'])}")
        return self.history
