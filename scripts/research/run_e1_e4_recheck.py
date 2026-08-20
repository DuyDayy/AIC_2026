"""
Kiểm lại E1 → E4 sau khi ASR được nối lại
==========================================

    .venv/bin/python scripts/run_e1_e4_recheck.py

VÌ SAO PHẢI CHẠY LẠI. `load_frame_ms()` đọc `data/Framme/*/metadata/*.csv`. Thư mục đó
bị xoá, hàm trả về `{}`, và `AsrSource` dùng bản đồ rỗng đó để gắn segment vào khung —
nên `covered` toàn False và ASR ghi **0 điểm cho mọi truy vấn**.

[ĐO] trước khi vá: ASR covered 0/173.426 khung, 0/20 truy vấn có điểm.
[ĐO] sau khi vá  : ASR covered 168.061/173.426 (96,9%), 20/20 truy vấn có điểm.

Nên mọi kết luận về ASR ở E1–E4 là hệ quả của một file thiếu, không phải phép đo:

  E1 kết luận "ASR chết hoàn toàn (R@100 = 0%)" và quy cho câu truy vấn có cụm
     "Tìm video quay cảnh…" nên BM25 không khớp. Lời giải thích đó KHÔNG THỂ đúng —
     không câu chữ nào khớp được với một nguồn phủ 0 khung.
  E2 kết luận expansion `qa` "vẫn 0%" — cùng nguyên nhân, không phải do expansion.
  E3 tính RRF với một nhánh toàn 0.
  E4 grid search thấy mọi trọng số ASR cho kết quả GIỐNG HỆT nhau tới 6 chữ số,
     đúng như một nhánh đóng góp bằng 0 phải thế.

Script này đo lại cả thang trên bench_kis (100 câu, cùng tập E1–E3 đã dùng) để các con
số so được trực tiếp với báo cáo cũ.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingestion.vector_index import load_flat_index
from src.retrieval.engine import SearchEngine
from src.retrieval.score_matrix import rrf_normalize
from src.retrieval.sources import (AsrSource, TextSource, VisualSource,
                                   load_asr_segments, load_frame_ms, load_ocr_text)
from src.scoring.harness import load_kis_queries
from src.scoring.rscore import K_THRESHOLDS, KISAnswer, final_score

RRF_K, BUDGET = 60.0, 100
OUT = Path("data/research/E4")


def expansion(path, queries):
    tbl = {q["id"]: q["query"] for q in json.load(open(path, encoding="utf-8"))["queries"]}
    return {q.query_id: tbl.get(q.query_id, q.query) for q in queries}


def metrics(queries, preds):
    acc = {k: 0.0 for k in K_THRESHOLDS}
    mrr = 0.0
    for q in queries:
        r = final_score(preds[q.query_id], q.ground_truth())
        mrr += r.mrr
        for k in K_THRESHOLDS:
            acc[k] += r.per_k[k]
    n = len(queries)
    return {"MRR": mrr / n, **{f"R@{k}": acc[k] / n for k in K_THRESHOLDS}}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    bench = load_kis_queries("data/eval/bench_kis_gt.json")
    bvecs = np.load("data/eval/bench_kis_gt_vecs.npy")

    idx = load_flat_index("data/embed")
    vis = VisualSource(idx)
    ocr = TextSource("ocr", idx.ids, load_ocr_text("data/OCR/ocr.jsonl"))
    asr = AsrSource(idx.ids, load_frame_ms(), load_asr_segments("data/ASR"))
    eng = SearchEngine(index=idx, visual=vis, text_sources=[ocr, asr],
                       weights={"visual": 1.0, "ocr": 1.0, "asr": 1.0}, mode="rrf")
    print(f"ASR covered {asr.score(bench[0].query).covered.mean():.1%} — "
          f"{'ĐÃ SỐNG' if asr.score(bench[0].query).covered.mean() > 0 else 'VẪN CHẾT'}",
          flush=True)

    qo = expansion("data/research/bench_kis_qo.json", bench)
    qa = expansion("data/research/bench_kis_qa.json", bench)

    print("tính trước 5 nhánh RRF…", flush=True)
    pre = {}
    for i, q in enumerate(bench):
        pre[q.query_id] = tuple(rrf_normalize(s.scores, s.covered, RRF_K) for s in (
            vis.score(bvecs[i]), ocr.score(q.query), ocr.score(qo[q.query_id]),
            asr.score(q.query), asr.score(qa[q.query_id])))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(bench)}", flush=True)

    def run(combine):
        return {q.query_id: [KISAnswer(a.video_id, a.frame_idx)
                             for a in eng.rank(combine(*pre[q.query_id]), BUDGET, dedup=True)]
                for q in bench}

    Z = np.zeros(idx.n_frames, dtype=np.float32)
    ladder = [
        ("E1 visual only",        lambda v, oo, oq, ao, aq: v),
        ("E1 ocr only",           lambda v, oo, oq, ao, aq: oo),
        ("E1 asr only",           lambda v, oo, oq, ao, aq: ao),
        ("E1 visual+ocr",         lambda v, oo, oq, ao, aq: v + oo),
        ("E1 visual+asr",         lambda v, oo, oq, ao, aq: v + ao),
        ("E1 all (equal RRF)",    lambda v, oo, oq, ao, aq: v + oo + ao),
        ("E2 ocr qo only",        lambda v, oo, oq, ao, aq: oq),
        ("E2 asr qa only",        lambda v, oo, oq, ao, aq: aq),
        ("E3 hierarchical RRF",   lambda v, oo, oq, ao, aq: v + (oo + oq) / 2 + (ao + aq) / 2),
        ("E3 hier (ocr QE only)", lambda v, oo, oq, ao, aq: v + (oo + oq) / 2 + ao),
    ]
    rows = []
    for name, fn in ladder:
        m = metrics(bench, run(fn))
        rows.append({"config": name, **m})
        print(f"  {name:<24} MRR {m['MRR']:.4f}  R@100 {m['R@100']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e1_e4_recheck_bench.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n✓ ghi {OUT}/e1_e4_recheck_bench.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
