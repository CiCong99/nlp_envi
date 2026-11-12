"""Cài đặt Transformer từ đầu (PyTorch thuần) cho 2 kiến trúc:

  - TransformerSeq2Seq : Encoder-Decoder (Vaswani et al., 2017)
  - GPTDecoder         : Decoder-only (kiểu GPT), dịch máy dưới dạng mô hình ngôn ngữ có điều kiện

Hai mô hình dùng CHUNG các khối (MultiHeadAttention, FeedForward, PositionalEncoding) để phép
so sánh kiến trúc là công bằng.

Quy ước mặt nạ (mask): tensor bool, True = ĐƯỢC PHÉP chú ý.
Luôn cho mỗi vị trí chú ý tới chính nó (đường chéo) để không có hàng nào bị che toàn bộ
-> tránh NaN khi câu bị đệm trái (left padding) lúc sinh văn bản với GPT.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Các khối dùng chung
# ----------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention:  Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + mask) V"""

    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        self.h, self.dk = n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = attn_dropout
        self.last_attn = None  # lưu trọng số attention khi cần trực quan hoá

    def forward(self, x_q, x_kv, mask=None, need_weights: bool = False):
        B, Lq, D = x_q.shape
        Lk = x_kv.shape[1]
        q = self.q_proj(x_q).view(B, Lq, self.h, self.dk).transpose(1, 2)   # (B,h,Lq,dk)
        k = self.k_proj(x_kv).view(B, Lk, self.h, self.dk).transpose(1, 2)
        v = self.v_proj(x_kv).view(B, Lk, self.h, self.dk).transpose(1, 2)
        p = self.attn_dropout if self.training else 0.0
        if need_weights:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
            if mask is not None:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            w = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            self.last_attn = w.detach()
            out = F.dropout(w, p, self.training) @ v
        else:
            # Cài đặt tối ưu (FlashAttention / memory-efficient) của PyTorch — kết quả tương đương
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        out = out.transpose(1, 2).reshape(B, Lq, D)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU() if activation == "relu" else nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """Mã hoá vị trí: 'sinusoidal' (cố định, như bài báo gốc) hoặc 'learned' (học được, như GPT)."""

    def __init__(self, d_model, max_len=1024, kind="sinusoidal"):
        super().__init__()
        self.kind = kind
        self.max_len = max_len
        if kind == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe, persistent=False)
        elif kind == "learned":
            self.emb = nn.Embedding(max_len, d_model)
            nn.init.normal_(self.emb.weight, std=0.02)
        else:
            raise ValueError(kind)

    def forward(self, position_ids):
        if int(position_ids.max()) >= self.max_len:
            raise ValueError(f"Chuỗi dài {int(position_ids.max()) + 1} vượt max_len={self.max_len}")
        return self.pe[position_ids] if self.kind == "sinusoidal" else self.emb(position_ids)


class Block(nn.Module):
    """Một lớp Transformer: self-attention [+ cross-attention] + FFN, có residual và LayerNorm.

    pre_norm=True  : x = x + Sublayer(LN(x))  (ổn định hơn, ít phụ thuộc warmup)
    pre_norm=False : x = LN(x + Sublayer(x))  (Post-LN, đúng bài báo gốc)
    """

    def __init__(self, d_model, n_heads, d_ff, dropout, attn_dropout, cross: bool, pre_norm: bool,
                 activation="relu"):
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attn = MultiHeadAttention(d_model, n_heads, attn_dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, attn_dropout) if cross else None
        self.ln2 = nn.LayerNorm(d_model) if cross else None
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.ln3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _sub(self, x, ln, fn):
        if self.pre_norm:
            return x + self.drop(fn(ln(x)))
        return ln(x + self.drop(fn(x)))

    def forward(self, x, self_mask, memory=None, cross_mask=None, need_weights=False):
        x = self._sub(x, self.ln1, lambda h: self.self_attn(h, h, self_mask, need_weights))
        if self.cross_attn is not None:
            x = self._sub(x, self.ln2, lambda h: self.cross_attn(h, memory, cross_mask, need_weights))
        x = self._sub(x, self.ln3, self.ffn)
        return x


def _diag(L, device):
    return torch.eye(L, dtype=torch.bool, device=device)


def causal_mask(L, device):
    return torch.ones(L, L, dtype=torch.bool, device=device).tril()


def _init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ----------------------------------------------------------------------------
# 1. Encoder-Decoder
# ----------------------------------------------------------------------------

@dataclass
class Seq2SeqConfig:
    vocab_size: int = 16000
    d_model: int = 512
    n_heads: int = 4
    d_ff: int = 1024
    enc_layers: int = 6
    dec_layers: int = 6
    dropout: float = 0.3
    attn_dropout: float = 0.1
    activation: str = "relu"
    pre_norm: bool = True
    pos_encoding: str = "sinusoidal"
    max_len: int = 512
    tie_embeddings: bool = True
    pad_id: int = 0

    def to_dict(self):
        return asdict(self)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # Từ vựng chung Anh-Việt => dùng chung 1 bảng embedding cho encoder, decoder và lớp đầu ra
        self.embed = nn.Embedding(cfg.vocab_size, d, padding_idx=cfg.pad_id)
        nn.init.normal_(self.embed.weight, mean=0, std=d ** -0.5)
        with torch.no_grad():
            self.embed.weight[cfg.pad_id].zero_()
        self.pos = PositionalEncoding(d, cfg.max_len, cfg.pos_encoding)
        self.drop = nn.Dropout(cfg.dropout)
        blk = lambda cross: Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.attn_dropout, cross,
                                  cfg.pre_norm, cfg.activation)
        self.encoder = nn.ModuleList([blk(False) for _ in range(cfg.enc_layers)])
        self.decoder = nn.ModuleList([blk(True) for _ in range(cfg.dec_layers)])
        self.enc_ln = nn.LayerNorm(d) if cfg.pre_norm else nn.Identity()
        self.dec_ln = nn.LayerNorm(d) if cfg.pre_norm else nn.Identity()
        if not cfg.tie_embeddings:
            self.out_proj = nn.Linear(d, cfg.vocab_size, bias=False)
        _init_weights(self.encoder)
        _init_weights(self.decoder)

    def _embed(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0).expand_as(ids)
        return self.drop(self.embed(ids) * math.sqrt(self.cfg.d_model) + self.pos(pos))

    def encode(self, src, need_weights=False):
        key_ok = src != self.cfg.pad_id                                   # (B,Ls)
        self_mask = key_ok[:, None, None, :] | _diag(src.size(1), src.device)
        x = self._embed(src)
        for layer in self.encoder:
            x = layer(x, self_mask, need_weights=need_weights)
        return self.enc_ln(x), key_ok

    def decode(self, tgt_in, memory, src_key_ok, need_weights=False):
        Lt = tgt_in.size(1)
        tgt_ok = tgt_in != self.cfg.pad_id
        self_mask = (causal_mask(Lt, tgt_in.device)[None, None] & tgt_ok[:, None, None, :]) | _diag(Lt, tgt_in.device)
        cross_mask = src_key_ok[:, None, None, :]
        x = self._embed(tgt_in)
        for layer in self.decoder:
            x = layer(x, self_mask, memory, cross_mask, need_weights=need_weights)
        x = self.dec_ln(x)
        if self.cfg.tie_embeddings:
            return F.linear(x, self.embed.weight)
        return self.out_proj(x)

    def forward(self, src, tgt_in):
        memory, key_ok = self.encode(src)
        return self.decode(tgt_in, memory, key_ok)


# ----------------------------------------------------------------------------
# 2. Decoder-only (GPT)
# ----------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 16000
    d_model: int = 512
    n_heads: int = 4
    d_ff: int = 1024
    n_layers: int = 14
    dropout: float = 0.3
    attn_dropout: float = 0.1
    activation: str = "relu"
    pre_norm: bool = True
    pos_encoding: str = "learned"
    max_len: int = 512
    tie_embeddings: bool = True
    pad_id: int = 0

    def to_dict(self):
        return asdict(self)


class GPTDecoder(nn.Module):
    """Chuỗi đầu vào: <s> câu_nguồn <sep> câu_đích </s>.  Che tương lai (causal mask)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, d, padding_idx=cfg.pad_id)
        nn.init.normal_(self.embed.weight, mean=0, std=d ** -0.5)
        with torch.no_grad():
            self.embed.weight[cfg.pad_id].zero_()
        self.pos = PositionalEncoding(d, cfg.max_len, cfg.pos_encoding)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([
            Block(d, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.attn_dropout, False, cfg.pre_norm, cfg.activation)
            for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(d) if cfg.pre_norm else nn.Identity()
        if not cfg.tie_embeddings:
            self.out_proj = nn.Linear(d, cfg.vocab_size, bias=False)
        _init_weights(self.layers)

    @staticmethod
    def position_ids_from_mask(attn_mask):
        """Vị trí tính từ token thật đầu tiên => đệm trái hay phải đều cho cùng vị trí."""
        return (attn_mask.long().cumsum(-1) - 1).clamp(min=0)

    def forward(self, input_ids, attn_mask=None, need_weights=False):
        B, L = input_ids.shape
        if attn_mask is None:
            attn_mask = input_ids != self.cfg.pad_id
        pos = self.position_ids_from_mask(attn_mask)
        x = self.drop(self.embed(input_ids) * math.sqrt(self.cfg.d_model) + self.pos(pos))
        mask = (causal_mask(L, input_ids.device)[None, None] & attn_mask[:, None, None, :]) | _diag(L, input_ids.device)
        for layer in self.layers:
            x = layer(x, mask, need_weights=need_weights)
        x = self.ln_f(x)
        if self.cfg.tie_embeddings:
            return F.linear(x, self.embed.weight)
        return self.out_proj(x)


# ----------------------------------------------------------------------------
# Tiện ích
# ----------------------------------------------------------------------------

def gpt_layers_to_match(target_params: int, base: GPTConfig, max_layers: int = 40) -> int:
    """Chọn số lớp GPT để tổng tham số gần nhất với mô hình Encoder-Decoder (so sánh công bằng)."""
    best, best_diff = 1, None
    for n in range(1, max_layers + 1):
        cfg = GPTConfig(**{**base.to_dict(), "n_layers": n})
        p = sum(x.numel() for x in GPTDecoder(cfg).parameters())
        diff = abs(p - target_params)
        if best_diff is None or diff < best_diff:
            best, best_diff = n, diff
    return best


def label_smoothed_ce(logits, target, eps: float = 0.1, ignore_index: int = -100):
    """Cross-entropy có label smoothing (Szegedy et al., 2016), trung bình trên các token hợp lệ.

    Trả về (loss_trung_bình, nll_trung_bình, số_token).  nll dùng để tính perplexity.
    """
    V = logits.size(-1)
    logits = logits.reshape(-1, V).float()
    target = target.reshape(-1)
    valid = target != ignore_index
    n = int(valid.sum())
    if n == 0:
        z = logits.sum() * 0.0
        return z, z.detach(), 0
    lp = torch.log_softmax(logits[valid], dim=-1)
    t = target[valid]
    nll = -lp.gather(1, t.unsqueeze(1)).squeeze(1)
    smooth = -lp.mean(dim=-1)
    loss = ((1 - eps) * nll + eps * smooth).mean()
    return loss, nll.mean().detach(), n
