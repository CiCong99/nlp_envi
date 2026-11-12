"""Tính COMET và BERTScore — chạy trong MÔI TRƯỜNG PYTHON RIÊNG (eval_env).

Lý do tách riêng: gói `unbabel-comet` yêu cầu numpy<2 và transformers<5; cài vào môi trường chính
của Colab sẽ hạ cấp thư viện và làm hỏng các notebook khác.

Script này KHÔNG import envi_lib, chỉ dùng thư viện chuẩn + comet + bert_score.
Kết quả được lưu đệm theo mã băm nội dung file => chạy lại chỉ tính hệ thống mới/thay đổi.

Cách dùng:
  python eval_neural.py --src test.en --ref test.vi --hyp a.vi.txt b.vi.txt --out neural.json [--no-comet] [--no-bertscore]
"""
import argparse
import hashlib
import json
import os
import sys
import unicodedata


def read(path):
    with open(path, encoding="utf-8") as f:
        return [unicodedata.normalize("NFC", l.rstrip("\n")) for l in f]


def sha1(lines):
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--hyp", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--bertscore-model", default="xlm-roberta-large")
    ap.add_argument("--bertscore-layers", type=int, default=None, help="chỉ cần khi dùng mô hình ngoài danh sách của bert_score")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--no-bertscore", action="store_true")
    a = ap.parse_args()

    import torch
    src, ref = read(a.src), read(a.ref)
    assert len(src) == len(ref), "src/ref lệch số dòng"
    results = {}
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            results = json.load(f)

    systems = {}
    for p in a.hyp:
        name = os.path.basename(p).replace(".vi.txt", "")
        hyp = read(p)
        if len(hyp) != len(ref):
            print(f"!! Bỏ qua {name}: {len(hyp)} dòng != {len(ref)}", flush=True)
            continue
        systems[name] = hyp

    def save():
        tmp = a.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
        os.replace(tmp, a.out)

    gpus = 1 if torch.cuda.is_available() else 0

    if not a.no_comet:
        todo = {n: h for n, h in systems.items()
                if results.get(n, {}).get("comet_hash") != sha1(h) + a.comet_model}
        if todo:
            from comet import download_model, load_from_checkpoint
            print(f"COMET ({a.comet_model}) cho {len(todo)} hệ thống ...", flush=True)
            model = load_from_checkpoint(download_model(a.comet_model))
            for n, h in todo.items():
                data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(src, h, ref)]
                o = model.predict(data, batch_size=a.batch, gpus=gpus, progress_bar=False)
                r = results.setdefault(n, {})
                r["COMET"] = round(100 * float(o.system_score), 2)
                r["COMET_seg"] = [round(float(x), 4) for x in o.scores]
                r["comet_model"] = a.comet_model
                r["comet_hash"] = sha1(h) + a.comet_model
                print(f"  {n}: COMET = {r['COMET']}", flush=True)
                save()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not a.no_bertscore:
        todo = {n: h for n, h in systems.items()
                if results.get(n, {}).get("bs_hash") != sha1(h) + a.bertscore_model}
        if todo:
            from bert_score import BERTScorer
            print(f"BERTScore ({a.bertscore_model}) cho {len(todo)} hệ thống ...", flush=True)
            scorer = BERTScorer(model_type=a.bertscore_model, num_layers=a.bertscore_layers, lang="vi",
                                batch_size=a.batch, device="cuda" if gpus else "cpu")
            for n, h in todo.items():
                # câu rỗng làm bert_score lỗi ở một số phiên bản => thay bằng "." (điểm sẽ thấp, đúng bản chất)
                hh = [x if x.strip() else "." for x in h]
                P, R, F = scorer.score(hh, ref)
                r = results.setdefault(n, {})
                r["BERTScore_P"] = round(100 * float(P.mean()), 2)
                r["BERTScore_R"] = round(100 * float(R.mean()), 2)
                r["BERTScore_F1"] = round(100 * float(F.mean()), 2)
                r["BERTScore_seg"] = [round(float(x), 4) for x in F.tolist()]
                r["bertscore_model"] = a.bertscore_model
                r["bs_hash"] = sha1(h) + a.bertscore_model
                print(f"  {n}: BERTScore F1 = {r['BERTScore_F1']}", flush=True)
                save()
    save()
    print("XONG", flush=True)


if __name__ == "__main__":
    sys.exit(main())
