"""Giải mã (sinh bản dịch) cho 2 mô hình tự cài: greedy và beam search.

beam_size = 1 trong beam search phải cho kết quả GIỐNG HỆT greedy — đây là một bài kiểm thử
tự động trong selftest.py.
"""
from __future__ import annotations

import torch

from .common import autocast_ctx
from .data import BOS_ID, EOS_ID, PAD_ID, gpt_prompt_batch, seq2seq_collate, sorted_eval_batches


# ----------------------------------------------------------------------------
# Lõi beam search dùng chung
# ----------------------------------------------------------------------------

@torch.no_grad()
def _beam_core(next_logprobs, batch_size, beam_size, max_new_tokens, device,
               eos_id=EOS_ID, pad_id=PAD_ID, length_penalty=1.0, min_new_tokens=1):
    """next_logprobs(gen: LongTensor (B*K, t)) -> FloatTensor (B*K, V) log-xác suất token kế tiếp.

    gen là phần ĐÃ SINH (không gồm prompt); hàm next_logprobs tự ghép với prompt/memory.
    Điểm chuẩn hoá theo độ dài: score / len^length_penalty (như fairseq/HF).
    """
    B, K = batch_size, beam_size
    beam_scores = torch.zeros(B, K, device=device)
    beam_scores[:, 1:] = float("-inf")          # bước đầu: chỉ mở rộng từ beam 0
    gen = torch.empty(B * K, 0, dtype=torch.long, device=device)
    finished = [[] for _ in range(B)]           # (điểm_chuẩn_hoá, list token)
    done = [False] * B

    for step in range(max_new_tokens):
        lp = next_logprobs(gen).float()          # (B*K, V)
        V = lp.size(-1)
        if step < min_new_tokens:
            lp[:, eos_id] = float("-inf")
        lp[:, pad_id] = float("-inf")
        cand = (beam_scores.view(-1, 1) + lp).view(B, K * V)
        top_s, top_i = cand.topk(2 * K, dim=-1)  # lấy 2K để còn đủ K beam sau khi tách beam kết thúc
        top_beam, top_tok = top_i // V, top_i % V

        new_scores = torch.full((B, K), float("-inf"), device=device)
        new_src = torch.zeros(B, K, dtype=torch.long, device=device)
        new_tok = torch.full((B, K), pad_id, dtype=torch.long, device=device)
        top_s_l, top_beam_l, top_tok_l = top_s.tolist(), top_beam.tolist(), top_tok.tolist()
        for b in range(B):
            if done[b]:
                new_src[b] = torch.arange(K, device=device)   # giữ nguyên, điền pad
                continue
            j = 0
            for s, kb, tok in zip(top_s_l[b], top_beam_l[b], top_tok_l[b]):
                if s == float("-inf"):
                    break
                if tok == eos_id:
                    seq = gen[b * K + kb].tolist()
                    finished[b].append((s / ((step + 1) ** length_penalty), seq))
                else:
                    new_scores[b, j] = s
                    new_src[b, j] = kb
                    new_tok[b, j] = tok
                    j += 1
                if j == K:
                    break
            # Điều kiện dừng (heuristic giống HF early_stopping=False)
            if len(finished[b]) >= K:
                finished[b].sort(key=lambda x: -x[0])
                finished[b] = finished[b][:K]
                best_active = new_scores[b, 0].item() / ((step + 1) ** length_penalty)
                if best_active <= finished[b][-1][0]:
                    done[b] = True
        if all(done):
            break
        flat_src = (torch.arange(B, device=device).unsqueeze(1) * K + new_src).view(-1)
        gen = torch.cat([gen[flat_src], new_tok.view(-1, 1)], dim=1)
        beam_scores = new_scores

    results = []
    for b in range(B):
        if not done[b]:
            for k in range(K):
                s = beam_scores[b, k].item()
                if s != float("-inf"):
                    finished[b].append((s / (gen.size(1) ** length_penalty), gen[b * K + k].tolist()))
        best = max(finished[b], key=lambda x: x[0])[1] if finished[b] else []
        results.append([t for t in best if t != pad_id])
    return results


@torch.no_grad()
def _greedy_core(next_logprobs, batch_size, max_new_tokens, device, eos_id=EOS_ID, pad_id=PAD_ID,
                 min_new_tokens=1):
    gen = torch.empty(batch_size, 0, dtype=torch.long, device=device)
    alive = torch.ones(batch_size, dtype=torch.bool, device=device)
    for step in range(max_new_tokens):
        lp = next_logprobs(gen).float()
        if step < min_new_tokens:
            lp[:, eos_id] = float("-inf")
        lp[:, pad_id] = float("-inf")
        tok = lp.argmax(-1)
        tok = torch.where(alive, tok, torch.full_like(tok, pad_id))
        gen = torch.cat([gen, tok.unsqueeze(1)], dim=1)
        alive &= tok != eos_id
        if not alive.any():
            break
    out = []
    for row in gen.tolist():
        seq = []
        for t in row:
            if t == eos_id or t == pad_id:
                break
            seq.append(t)
        out.append(seq)
    return out


def _force_eos_after_limit(lp, gen_len, limits, eos_id=EOS_ID):
    """Mỗi câu có giới hạn độ dài riêng (theo độ dài câu nguồn): quá giới hạn thì buộc sinh </s>.

    Nhờ vậy kết quả dịch một câu KHÔNG phụ thuộc việc nó nằm chung batch với câu dài hay ngắn.
    """
    over = limits[: lp.size(0)] <= gen_len
    if over.any():
        lp[over] = float("-inf")
        lp[over, eos_id] = 0.0
    return lp


# ----------------------------------------------------------------------------
# Encoder-Decoder
# ----------------------------------------------------------------------------

@torch.no_grad()
def seq2seq_generate_batch(model, src, beam_size=1, max_len_a=1.5, max_len_b=10, length_penalty=1.0,
                           amp_dtype=None):
    """src: (B, Ls) đã có </s>, đệm phải.  Trả về list token id (không gồm </s>)."""
    model.eval()
    device = src.device
    B = src.size(0)
    src_lens = (src != PAD_ID).sum(1)
    limits = (src_lens.float() * max_len_a + max_len_b).long().clamp(max=model.cfg.max_len - 2)
    max_new = int(limits.max()) + 1
    with autocast_ctx(amp_dtype):
        memory, key_ok = model.encode(src)
    K = beam_size
    if K > 1:
        memory = memory.repeat_interleave(K, dim=0)
        key_ok = key_ok.repeat_interleave(K, dim=0)
        limits = limits.repeat_interleave(K, dim=0)
    bos = torch.full((B * K, 1), BOS_ID, dtype=torch.long, device=device)

    def next_lp(gen):
        tgt = torch.cat([bos[: gen.size(0)], gen], dim=1)
        with autocast_ctx(amp_dtype):
            logits = model.decode(tgt, memory[: gen.size(0)], key_ok[: gen.size(0)])
        lp = torch.log_softmax(logits[:, -1].float(), dim=-1)
        return _force_eos_after_limit(lp, gen.size(1), limits)

    if K == 1:
        return _greedy_core(next_lp, B, max_new, device)
    return _beam_core(next_lp, B, K, max_new, device, length_penalty=length_penalty)


# ----------------------------------------------------------------------------
# Decoder-only (GPT)
# ----------------------------------------------------------------------------

@torch.no_grad()
def gpt_generate_batch(model, src_ids_list, sep_id, beam_size=1, max_len_a=1.5, max_len_b=10,
                       length_penalty=1.0, amp_dtype=None, device=None):
    model.eval()
    device = device or next(model.parameters()).device
    prompt, pmask = gpt_prompt_batch(src_ids_list, sep=sep_id)
    prompt, pmask = prompt.to(device), pmask.to(device)
    B = prompt.size(0)
    room = model.cfg.max_len - prompt.size(1) - 1
    if room <= 1:
        raise ValueError("Câu nguồn quá dài so với max_len của GPT.")
    limits = torch.tensor([min(int(len(s) * max_len_a + max_len_b), room) for s in src_ids_list], device=device)
    max_new = int(limits.max()) + 1
    K = beam_size
    if K > 1:
        prompt = prompt.repeat_interleave(K, dim=0)
        pmask = pmask.repeat_interleave(K, dim=0)
        limits = limits.repeat_interleave(K, dim=0)

    def next_lp(gen):
        n = gen.size(0)
        ids = torch.cat([prompt[:n], gen], dim=1)
        mask = torch.cat([pmask[:n], torch.ones_like(gen, dtype=torch.bool)], dim=1)
        with autocast_ctx(amp_dtype):
            logits = model(ids, mask)
        lp = torch.log_softmax(logits[:, -1].float(), dim=-1)
        return _force_eos_after_limit(lp, gen.size(1), limits)

    if K == 1:
        return _greedy_core(next_lp, B, max_new, device)
    return _beam_core(next_lp, B, K, max_new, device, length_penalty=length_penalty)


# ----------------------------------------------------------------------------
# Dịch cả một tập câu (có sắp xếp theo độ dài để nhanh, rồi trả về đúng thứ tự gốc)
# ----------------------------------------------------------------------------

def translate_corpus(model, tokenizer, src_texts, kind: str, beam_size=1, batch_size=64,
                     amp_dtype=None, length_penalty=1.0, max_src_tokens=None, progress=True):
    """kind: 'seq2seq' hoặc 'gpt'.  Trả về list câu tiếng Việt cùng thứ tự với src_texts."""
    from tqdm.auto import tqdm
    device = next(model.parameters()).device
    src_ids = tokenizer.encode(src_texts)
    if max_src_tokens:
        n_trunc = sum(1 for s in src_ids if len(s) > max_src_tokens)
        if n_trunc:
            print(f"⚠️  {n_trunc} câu nguồn dài hơn {max_src_tokens} token bị cắt bớt khi dịch.")
        src_ids = [s[:max_src_tokens] for s in src_ids]
    bs = max(1, batch_size // beam_size) if beam_size > 1 else batch_size
    batches = sorted_eval_batches([len(s) for s in src_ids], bs)
    out = [None] * len(src_ids)
    it = tqdm(batches, desc=f"Dịch ({kind}, beam={beam_size})", leave=False) if progress else batches
    for idx in it:
        if kind == "seq2seq":
            batch = seq2seq_collate([(i, src_ids[i], None) for i in idx])
            ids = seq2seq_generate_batch(model, batch["src"].to(device), beam_size=beam_size,
                                         length_penalty=length_penalty, amp_dtype=amp_dtype)
        elif kind == "gpt":
            ids = gpt_generate_batch(model, [src_ids[i] for i in idx], tokenizer.sep_id, beam_size=beam_size,
                                     length_penalty=length_penalty, amp_dtype=amp_dtype, device=device)
        else:
            raise ValueError(kind)
        for i, h in zip(idx, tokenizer.decode(ids)):
            out[i] = h
    assert all(o is not None for o in out)
    return out
