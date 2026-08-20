"""
E4 — Global Weighted RRF, tune hai tầng (beta expansion → alpha modality)
========================================================================

    .venv/bin/python scripts/run_e4_wrrf.py

E3 trả lời "bỏ phiếu ngang nhau có tốt không?". E4 hỏi "mỗi modality nên có bao nhiêu
quyền vote?" — và vì hệ đang dùng query expansion, quyền vote phải chia làm HAI tầng:

    Score(d) = Σ_m  alpha_m · ( Σ_j beta_mj / (k + rank_mj(d)) )
    ràng buộc: Σ alpha_m = 1 ,  Σ_j beta_mj = 1  cho từng m

Chuẩn hoá beta TRONG từng modality ngăn Visual tự động có nhiều quyền vote chỉ vì nó có
nhiều expansion hơn. Trọng số áp vào **đóng góp RRF**, không áp vào raw score — cosine và
BM25 không bao giờ được cộng trực tiếp với nhau.

=============================================================================
NHỮNG GÌ ĐƯỢC ĐÓNG BĂNG, VÀ VÌ SAO
=============================================================================

`k = 60`, độ sâu candidate, dedup, canonicalization: giữ Y HỆT E3. Thay nhiều nhóm biến
cùng lúc thì không biết gain đến từ đâu.

`beta_visual = [1.0]` — chỉ dùng câu gốc, KHÔNG expansion. Đây là kết quả đo của E2 chứ
không phải giả định: dịch sang tiếng Anh hạ R@100 từ 0,78 xuống 0,68; paraphrase tiếng
Việt xuống 0,77. Jina-CLIP-v2 xử lý tiếng Việt gốc tốt hơn mọi bản viết lại đã thử.

Tập tuning = gtv2 (100) + holdout (110). `queries_smoke` bị loại: nó là 3 câu CON của
gtv2, cùng `id`, nên gộp vào chỉ nhân ba câu đó lên gấp đôi.

Tập held-out = bench_kis (100), không đụng tới cho tới bước báo cáo cuối.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.research.run_e1_ab import load_asr_segments, load_frame_ms, load_ocr_text
from src.ingestion.vector_index import load_flat_index
from src.retrieval.engine import SearchEngine
from src.retrieval.score_matrix import rrf_normalize
from src.retrieval.sources import AsrSource, TextSource, VisualSource
from src.scoring.harness import load_kis_queries
from src.scoring.rscore import K_THRESHOLDS, KISAnswer, final_score

OUT = Path("data/research/E4")
RRF_K = 60.0
BUDGET = 100


def load_expansion(path: str, queries) -> dict[str, str]:
    """`{query_id: câu mở rộng}`; câu thiếu lùi về câu gốc.

    Lùi về câu gốc KHÔNG phải là trung tính: nhánh expansion khi đó trùng nhánh gốc, nên
    modality đó tự nhân đôi quyền vote của câu gốc. Hàm trả thêm số câu đã lùi để báo cáo
    nói rõ mức nhiễu này thay vì giấu nó.
    """
    with open(path, encoding="utf-8") as fh:
        table = {q["id"]: q["query"] for q in json.load(fh)["queries"]}
    return {q.query_id: table.get(q.query_id, q.query) for q in queries}


def metrics_of(queries, preds: dict[str, list]) -> dict[str, float]:
    """MRR + R@k. `K_THRESHOLDS` là (1,5,20,50,100) — KHÔNG có R@10.

    Bản E4 trước gán nhãn `per_k[20]` thành "R@10"; con số vẫn đúng nhưng tên sai, và một
    bảng dán nhãn sai thì tệ hơn một bảng thiếu cột.
    """
    acc = {k: 0.0 for k in K_THRESHOLDS}
    mrr = 0.0
    for q in queries:
        rep = final_score(preds[q.query_id], q.ground_truth())
        mrr += rep.mrr
        for k in K_THRESHOLDS:
            acc[k] += rep.per_k[k]
    n = len(queries)
    out = {"MRR": mrr / n}
    out.update({f"R@{k}": acc[k] / n for k in K_THRESHOLDS})
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    tune = load_kis_queries("data/eval/gtv2_gt.json") + load_kis_queries("data/eval/holdout_gt.json")
    held = load_kis_queries("data/eval/bench_kis_gt.json")
    tune_vecs = np.concatenate([np.load("data/eval/gtv2_vecs.npy"),
                                np.load("data/eval/holdout_vecs.npy")], axis=0)
    held_vecs = np.load("data/eval/bench_kis_gt_vecs.npy")
    queries = tune + held
    vecs = np.concatenate([tune_vecs, held_vecs], axis=0)
    qtype = {}
    for path in ("data/eval/gtv2_gt.json", "data/eval/holdout_gt.json"):
        for d in json.load(open(path, encoding="utf-8"))["queries"]:
            qtype[d["id"]] = d.get("query_type", "?")
    print(f"tuning {len(tune)} · held-out {len(held)} · tổng {len(queries)}", flush=True)

    qo = {}
    qa = {}
    for name, qs in (("gtv2", tune[:100]), ("holdout", tune[100:]), ("bench_kis", held)):
        base = "data/research/expansions" if name != "bench_kis" else "data/research"
        qo.update(load_expansion(f"{base}/{name}_qo.json", qs))
        qa.update(load_expansion(f"{base}/{name}_qa.json", qs))
    fell_back = sum(1 for q in queries if qo[q.query_id] == q.query)
    print(f"qo lùi về câu gốc: {fell_back}/{len(queries)} truy vấn", flush=True)

    index = load_flat_index("data/embed")
    visual = VisualSource(index)
    ocr = TextSource("ocr", index.ids, load_ocr_text("data/OCR/ocr.jsonl"))
    asr = AsrSource(index.ids, load_frame_ms(), load_asr_segments("data/ASR"))
    engine = SearchEngine(index=index, visual=visual, text_sources=[ocr, asr],
                          weights={"visual": 1.0, "ocr": 1.0, "asr": 1.0}, mode="rrf")

    print("tính trước RRF cho 5 nhánh…", flush=True)
    pre: dict[str, tuple] = {}
    for i, q in enumerate(queries):
        v = visual.score(vecs[i])
        oo, oq = ocr.score(q.query), ocr.score(qo[q.query_id])
        ao, aq = asr.score(q.query), asr.score(qa[q.query_id])
        pre[q.query_id] = tuple(
            rrf_normalize(s.scores, s.covered, RRF_K)
            for s in (v, oo, oq, ao, aq)
        )
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(queries)}", flush=True)

    def predict(qs, alpha, beta_o, beta_a):
        av, ao_, aa = alpha
        bo0, bo1 = beta_o
        ba0, ba1 = beta_a
        out = {}
        for q in qs:
            rv, roo, roq, rao, raq = pre[q.query_id]
            s = av * rv + ao_ * (bo0 * roo + bo1 * roq) + aa * (ba0 * rao + ba1 * raq)
            out[q.query_id] = [KISAnswer(a.video_id, a.frame_idx)
                               for a in engine.rank(s, top_k=BUDGET, dedup=True)]
        return out

    EQ = (1 / 3, 1 / 3, 1 / 3)

    # ── TẦNG 1: beta, với alpha ĐÓNG BĂNG ở equal (đúng cấu hình E3) ──────────
    print("\n[1/4] tầng 1 — beta expansion (alpha giữ equal, như E3)", flush=True)
    rows = []
    for b in np.arange(0.0, 1.0 + 1e-9, 0.1):
        b = round(float(b), 2)
        for tag, bo, ba in (("ocr", (b, 1 - b), (0.5, 0.5)), ("asr", (0.5, 0.5), (b, 1 - b))):
            m = metrics_of(tune, predict(tune, EQ, bo, ba))
            rows.append({"tier": "beta", "which": tag, "w_original": b,
                         "w_expansion": round(1 - b, 2), **m})
    beta_df = pd.DataFrame(rows)
    beta_df.to_csv(OUT / "e4_beta_search.csv", index=False)
    best_o = beta_df[beta_df.which == "ocr"].sort_values("MRR", ascending=False).iloc[0]
    best_a = beta_df[beta_df.which == "asr"].sort_values("MRR", ascending=False).iloc[0]
    BETA_O = (float(best_o.w_original), float(best_o.w_expansion))
    BETA_A = (float(best_a.w_original), float(best_a.w_expansion))
    print(f"  beta_OCR = {BETA_O}  MRR {best_o.MRR:.4f}")
    print(f"  beta_ASR = {BETA_A}  MRR {best_a.MRR:.4f}")

    # ── TẦNG 2: alpha trên simplex, beta ĐÓNG BĂNG ───────────────────────────
    def sweep(step):
        res = []
        grid = np.arange(0.0, 1.0 + 1e-9, step)
        for av in grid:
            for ao_ in grid:
                aa = 1.0 - av - ao_
                if aa < -1e-9 or aa > 1.0 + 1e-9:
                    continue
                a = (round(float(av), 3), round(float(ao_), 3), round(max(0.0, aa), 3))
                res.append({"w_visual": a[0], "w_ocr": a[1], "w_asr": a[2],
                            **metrics_of(tune, predict(tune, a, BETA_O, BETA_A))})
        return pd.DataFrame(res).sort_values("MRR", ascending=False)

    print("\n[2/4] tầng 2 — alpha, lưới thô bước 0,1", flush=True)
    coarse = sweep(0.1)
    coarse.to_csv(OUT / "e4_weight_search_coarse.csv", index=False)
    print(coarse.head(5).to_string(index=False))

    c = coarse.iloc[0]
    print("\n[3/4] tinh chỉnh bước 0,05 quanh vùng tốt", flush=True)
    fine = sweep(0.05)
    fine = fine[(abs(fine.w_visual - c.w_visual) <= 0.15)
                & (abs(fine.w_ocr - c.w_ocr) <= 0.15)].sort_values("MRR", ascending=False)
    fine.to_csv(OUT / "e4_weight_search.csv", index=False)
    print(fine.head(5).to_string(index=False))
    ALPHA = (float(fine.iloc[0].w_visual), float(fine.iloc[0].w_ocr), float(fine.iloc[0].w_asr))
    print(f"\nCHỐT: alpha={ALPHA} · beta_OCR={BETA_O} · beta_ASR={BETA_A}")

    # ── HELD-OUT: chỉ chạm tới ở đây ─────────────────────────────────────────
    print("\n[4/4] held-out (bench_kis, 100 câu) — lần đầu chạm tới", flush=True)
    e3_held = predict(held, EQ, (0.5, 0.5), (0.5, 0.5))
    e4_held = predict(held, ALPHA, BETA_O, BETA_A)
    m3, m4 = metrics_of(held, e3_held), metrics_of(held, e4_held)

    rng = np.random.default_rng(0)
    d = np.array([final_score(e4_held[q.query_id], q.ground_truth()).mrr
                  - final_score(e3_held[q.query_id], q.ground_truth()).mrr for q in held])
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])

    print(f"  {'':<10}" + "".join(f"{k:>10}" for k in m3))
    print(f"  {'E3 equal':<10}" + "".join(f"{m3[k]:>10.4f}" for k in m3))
    print(f"  {'E4 WRRF':<10}" + "".join(f"{m4[k]:>10.4f}" for k in m4))
    print(f"  {'Δ':<10}" + "".join(f"{m4[k] - m3[k]:>+10.4f}" for k in m3))
    print(f"\n  ΔMRR bắt cặp {d.mean():+.4f}  KTC95 [{lo:+.4f}, {hi:+.4f}]"
          f"  thắng {(d > 0).sum()} / thua {(d < 0).sum()}"
          f"  {'✓ E4 hơn, chắc' if lo > 0 else '✗ E4 KÉM HƠN, chắc' if hi < 0 else '— không phân biệt được'}")

    # ── leave-one-out trên tuning, để đối chiếu với alpha học được ────────────
    loo = []
    for tag, a in (("V", (1, 0, 0)), ("V+O", (0.5, 0.5, 0)), ("V+A", (0.5, 0, 0.5)),
                   ("O+A", (0, 0.5, 0.5)), ("V+O+A (alpha*)", ALPHA)):
        loo.append({"config": tag, **metrics_of(tune, predict(tune, a, BETA_O, BETA_A))})
    pd.DataFrame(loo).to_csv(OUT / "e4_leave_one_out.csv", index=False)
    print("\n── leave-one-out (tuning) ──")
    print(pd.DataFrame(loo).to_string(index=False))

    # ── theo loại đề, CHỈ trên tuning: bench_kis không có query_type ─────────
    e3_t, e4_t = predict(tune, EQ, (0.5, 0.5), (0.5, 0.5)), predict(tune, ALPHA, BETA_O, BETA_A)
    by = defaultdict(lambda: {"n": 0, "e3": 0.0, "e4": 0.0})
    for q in tune:
        b = by[qtype.get(q.query_id, "?")]
        b["n"] += 1
        b["e3"] += final_score(e3_t[q.query_id], q.ground_truth()).mrr
        b["e4"] += final_score(e4_t[q.query_id], q.ground_truth()).mrr
    trows = [{"query_type": t, "n": v["n"], "E3_MRR": v["e3"] / v["n"],
              "E4_MRR": v["e4"] / v["n"], "delta": (v["e4"] - v["e3"]) / v["n"]}
             for t, v in sorted(by.items())]
    pd.DataFrame(trows).to_csv(OUT / "e4_per_type.csv", index=False)
    print("\n── theo loại đề (TUNING; bench_kis không có query_type) ──")
    print(pd.DataFrame(trows).to_string(index=False))

    # ── per-query trên held-out, đầu vào cho E5 ──────────────────────────────
    pq = []
    for q in held:
        r3 = final_score(e3_held[q.query_id], q.ground_truth())
        r4 = final_score(e4_held[q.query_id], q.ground_truth())
        a, b = (r3.best_rank or 999), (r4.best_rank or 999)
        pq.append({"query_id": q.query_id, "query": q.query, "E3_rank": a,
                   "E4_rank": b, "delta": a - b})
    pd.DataFrame(pq).to_csv(OUT / "e4_per_query.csv", index=False)

    json.dump({"alpha": ALPHA, "beta_ocr": BETA_O, "beta_asr": BETA_A, "rrf_k": RRF_K,
               "tuning_n": len(tune), "heldout_n": len(held),
               "qo_fell_back": fell_back,
               "heldout": {"E3_equal": m3, "E4_wrrf": m4,
                           "delta_mrr_paired": float(d.mean()),
                           "ci95": [float(lo), float(hi)],
                           "win": int((d > 0).sum()), "loss": int((d < 0).sum())}},
              open(OUT / "summary.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n✓ đã ghi {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
