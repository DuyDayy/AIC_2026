"""
Chạy ĐÚNG kiến trúc thi trên bộ đề thử nghiệm của BTC → bài nộp chuẩn thể lệ
============================================================================

    ① probe + mở rộng  →  ② 7 run  →  ③ RRF hai tầng (ĐỀU)  →  ⑤a rổ
                                                                  │
                    ⑦ CSV + zip  ←  ④ DANTE  ←  ⑤c rerank (API)  ←┘

Giống `scripts/run.py` ở MỌI tầng tính điểm — cùng `hierarchical_rrf`, cùng
`fused_pool`, cùng `k_best_alignments`, cùng `write_task_csv`. Khác đúng hai chỗ,
và cả hai đều là chỗ ĐANG THIẾU HẠ TẦNG chứ không phải chỗ đổi thiết kế:

  · ⑤c rerank: thay Qwen2.5-VL bằng API của đội. KHÔNG phải vì API tốt hơn — mà
    vì bộ keyframe không có trên đĩa, nên đường đọc ảnh của ⑤c không chạy được.
    Đây là rerank THEO VĂN BẢN (OCR + lời dẫn quanh khung), tức nó chỉ thấy được
    phần bằng chứng chữ. Mọi câu phân biệt bằng hình sẽ không được nó giúp.
  · mã hoá truy vấn chạy tại máy (MPS) thay vì Modal, có ghim revision.

λ = 0,0 — giá trị đã quét (`src/retrieval/dante.py`); mọi λ > 0 đều tệ hơn vì
`−λ(t−τ)` phạt theo khoảng cách tuyệt đối nên nó can thiệp vào XẾP HẠNG VIDEO.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.jina_encoder import truncate_and_normalize      # noqa: E402
from src.ingestion.vector_index import load_flat_index             # noqa: E402
from src.retrieval.dante import DEFAULT_LAMBDA                     # noqa: E402
from src.retrieval.pool import fused_pool                          # noqa: E402
from src.retrieval.probe import build_probes, declarativize        # noqa: E402
from src.retrieval.score_matrix import (                           # noqa: E402
    SourceScores, hierarchical_rrf,
)
from src.retrieval.sources import (                                # noqa: E402
    AsrSource, TextSource, VisualSource, load_asr_segments, load_frame_ms,
    load_ocr_text,
)
from src.submission.kbest import k_best_alignments                 # noqa: E402
from src.submission.writer import (                                # noqa: E402
    TaskSubmission, pack_submission_zip, task_type_from_filename, write_task_csv,
)

# ── kiến trúc: ĐỀU ở cả hai tầng ────────────────────────────────────────────
ALPHA = {"visual": 1 / 3, "ocr": 1 / 3, "asr": 1 / 3}
RRF_K = 60.0
POOL_CAP = 300
TOP_K = 100
TRAKE_K_PER_VIDEO = 1
MODEL_SHA = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
CODE_SHA = "39e6a55ae971b59bea6e44675d237c99762e7ee2"

#: ⑤c hợp vào nền bằng RRF TẦNG BA. `0,25` là giá trị MANG SANG: nó quét ra khi ⑤
#: còn hợp bằng chuẩn hoá z, và nay khác đơn vị với thứ nó được đo. Chưa quét lại.
RERANK_W = 0.25
RERANK_TOP = 60          #: chấm lại bao nhiêu ứng viên đầu rổ
RERANK_BATCH = 10        #: mỗi lời gọi API chấm bao nhiêu khung


def _beta_for(n: int) -> tuple[float, ...]:
    """ĐỀU — mỗi bản mở rộng đóng góp ngang câu gốc."""
    return (1.0 / n,) * max(n, 1)


def nap_de(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.glob("*.txt")):
        t = p.read_text(encoding="utf-8").strip()
        if t:
            out.append({"id": p.stem, "kind": task_type_from_filename(p.stem), "text": t})
    return out


#: Tag trong tên file → tên modality. `vis` là bản mô tả cảnh bằng tiếng Anh, đi vào
#: nguồn THỊ GIÁC (phải mã hoá bằng jina-clip, không phải BM25 như ocr/asr).
TAG_MOD = {"ocr": "ocr", "asr": "asr", "vis": "visual", "visual": "visual"}


def nap_mo_rong(d: Path) -> dict[str, dict[str, list[str]]]:
    """
    `expansions/<id>.<tag><n>.txt` → `{id: {visual: [...], ocr: [...], asr: [...]}}`.

    🔴 Bản trước CHỈ nhận `ocr`/`asr`, nên các bản mở rộng THỊ GIÁC đã sinh ra thì
    nằm im trên đĩa. Đó là thiệt hại thật: [ĐO] thị giác là nguồn đơn MẠNH NHẤT
    (R@100 25,3% so với OCR 16,5% và ASR 19,3%) mà lại chạy đúng MỘT run, trong khi
    hai nguồn yếu hơn mỗi bên ba run.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for f in sorted(d.glob("*.txt")):
        stem, _, tag = f.stem.rpartition(".")
        mod = TAG_MOD.get(tag.rstrip("0123456789"))
        if not stem or mod is None:
            continue
        t = f.read_text(encoding="utf-8").strip()
        if t:
            out.setdefault(stem, {}).setdefault(mod, []).append(t)
    return out


# ============================================================
# ⑤c RERANK BẰNG API — thay thế tạm cho Qwen2.5-VL
# ============================================================


def rerank_api(jobs: list[dict], model: str, workers: int = 6) -> dict[str, float]:
    """
    Chấm lại độ khớp của từng khung bằng API, trả `{khoá: điểm 0–1}`.

    Gửi theo LÔ `RERANK_BATCH` khung một lời gọi, và bắt model trả JSON thuần
    `{"<khoá>": <điểm>}`. Gửi từng khung một sẽ tốn 60 lời gọi mỗi truy vấn.

    Khung nào API không chấm (lỗi mạng, JSON hỏng, thiếu khoá) thì KHÔNG có mặt
    trong dict trả về — bên gọi coi nó là "không phủ" chứ không gán 0. Gán 0 cho
    một khung chỉ vì API rớt gói là phạt nó ngang với "chắc chắn sai".
    """
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    client = OpenAI(base_url="https://sv.devquote.shop/v1",
                    api_key=os.environ.get("MIRAI_API_KEY", "") or "dummy",
                    timeout=90.0, max_retries=2)

    HE = ("Bạn chấm độ khớp giữa một TRUY VẤN tìm cảnh trong video tin tức tiếng Việt "
          "và các KHUNG ứng viên. Mỗi khung chỉ được mô tả bằng chữ trên màn hình (OCR) "
          "và lời dẫn (ASR) quanh thời điểm đó — bạn KHÔNG thấy hình. "
          "Cho mỗi khung một điểm 0–100: 0 = chắc chắn không liên quan, "
          "100 = gần như chắc chắn đúng cảnh được mô tả. "
          "Thiếu bằng chứng chữ thì cho điểm giữa (40–60), ĐỪNG cho 0. "
          'Trả về DUY NHẤT một object JSON dạng {"khoá": điểm}, không giải thích.')

    out: dict[str, float] = {}

    def mot_lo(lo):
        mo = "\n\n".join(
            f'[{j["key"]}]\nOCR: {j["ocr"][:300] or "(trống)"}\n'
            f'LỜI DẪN: {j["asr"][:300] or "(trống)"}' for j in lo)
        msg = f"TRUY VẤN:\n{lo[0]['query']}\n\nCÁC KHUNG:\n{mo}"
        for _ in range(2):
            try:
                r = client.chat.completions.create(
                    model=model, max_tokens=2048,
                    messages=[{"role": "system", "content": HE},
                              {"role": "user", "content": msg}])
                t = (r.choices[0].message.content or "").strip()
                if "```" in t:
                    t = t.split("```")[1].lstrip("json").strip()
                d = json.loads(t[t.index("{"):t.rindex("}") + 1])
                return {k: max(0.0, min(1.0, float(v) / 100.0)) for k, v in d.items()}
            except Exception:
                continue
        return {}

    los = [jobs[i:i + RERANK_BATCH] for i in range(0, len(jobs), RERANK_BATCH)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d in ex.map(mot_lo, los):
            out.update(d)
    return out


# ============================================================
# CHẠY
# ============================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="THUNGHIEM-bo-de-thi")
    ap.add_argument("--expansions", default="expansions")
    ap.add_argument("--out", default="submission_thunghiem")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    Q = nap_de(Path(a.dir))
    EXP = nap_mo_rong(Path(a.expansions))
    print(f"① {len(Q)} đề: "
          + " · ".join(f"{k} {sum(1 for q in Q if q['kind'] == k)}"
                       for k in ("kis", "qa", "trake")))

    # ① probe. Q&A đi qua `declarativize` — câu hỏi thành câu MÔ TẢ, vì chỉ mục là
    # ảnh: "màu áo của người dẫn là gì" không giống bất cứ khung nào, "người dẫn mặc
    # áo" thì có.
    for q in Q:
        src = declarativize(q["text"]) if q["kind"] == "qa" else q["text"]
        q["probes"] = [p.text for p in build_probes(src)] or [src]
        # KIS/QA nộp MỘT khung mỗi dòng ⟹ ép về một hàng ở ③ (hậu tố tên file quyết
        # định số cột, phép tách mốc KHÔNG được sửa thể lệ).
        q["gop"] = q["kind"] in ("kis", "qa")
    can = sorted({t for q in Q for t in q["probes"]}
                 | {x for q in Q for v in EXP.get(q["id"], {}).values() for x in v})
    print(f"  {len(can)} đoạn cần vector · mở rộng phủ {len(EXP)}/{len(Q)} đề")

    # ── mã hoá tại máy, GHIM revision ──────────────────────────────────────
    import hashlib
    cache_p = Path("data/embed/query_cache.npz")
    cache = dict(np.load(cache_p)) if cache_p.is_file() else {}
    key = lambda t: hashlib.sha1(t.encode()).hexdigest()[:16]      # noqa: E731
    thieu = [t for t in can if key(t) not in cache]
    if thieu:
        import torch
        from transformers import AutoModel
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"  mã hoá {len(thieu)} đoạn trên {dev} @ {MODEL_SHA[:8]}…", flush=True)
        m = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True,
                                      revision=MODEL_SHA, code_revision=CODE_SHA).eval().to(dev)
        for i in range(0, len(thieu), 32):
            lo = thieu[i:i + 32]
            with torch.inference_mode():
                v = np.asarray(m.encode_text(lo, batch_size=32), dtype=np.float32)
            v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            for t, x in zip(lo, v):
                cache[key(t)] = x
        np.savez(cache_p, **cache)
        del m
    QV = {t: truncate_and_normalize(np.asarray([cache[key(t)]], np.float32), a.dim)[0]
          for t in can}

    idx = load_flat_index("data/embed", dim=a.dim)
    FI = np.asarray(idx.frame_idx)
    VID = np.array([v for v, _ in idx.ids])
    fms = load_frame_ms()
    TMS = np.array([fms.get((v, int(n)), -1.0) for v, n in idx.ids], dtype=np.float64)
    vis = VisualSource(idx)
    ocr_map = load_ocr_text("data/OCR/ocr.jsonl")
    asr_seg = load_asr_segments("data/ASR")
    tsrc = [TextSource("ocr", idx.ids, ocr_map), AsrSource(idx.ids, fms, asr_seg)]
    print(f"① xong {time.time() - t0:.0f}s · {idx.n_frames:,} khung · {len(idx.ranges)} video")

    # ── ②③ 7 run, RRF hai tầng, ĐỀU cả hai tầng ────────────────────────────
    n_run = []
    for i, q in enumerate(Q):
        rows = []
        for t in q["probes"]:
            runs = {"visual": [vis.score(QV[t])]}
            for s in tsrc:
                mr = EXP.get(q["id"], {}).get(s.name) or []
                runs[s.name] = [s.score(t)] + [s.score(x) for x in mr]
            beta = {m: _beta_for(len(r)) for m, r in runs.items()}
            rows.append(hierarchical_rrf(runs, alpha=ALPHA, beta=beta, k=RRF_K))
            n_run.append(sum(len(r) for r in runs.values()))
        S = np.vstack(rows).astype(np.float32)
        if q["gop"] and S.shape[0] > 1:
            S = S.max(axis=0, keepdims=True)
        q["S"] = S
        print(f"  ②③ {i + 1}/{len(Q)} {q['id']}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"②③ xong {time.time() - t0:.0f}s · {int(np.median(n_run))} run/probe "
          f"· alpha ĐỀU 1/3 · beta ĐỀU 1/{int(np.median(n_run) // 3) or 1}")

    # ── ⑤a rổ ──────────────────────────────────────────────────────────────
    for q in Q:
        q["pool"] = fused_pool(q["S"], POOL_CAP)
    print(f"⑤a rổ {int(np.median([len(q['pool']) for q in Q]))} khung/truy vấn")

    # ── ⑤c rerank bằng API, hợp vào nền bằng RRF TẦNG BA ───────────────────
    if not a.no_rerank:
        jobs, span = [], {}
        for q in Q:
            base = q["S"].max(axis=0)
            cand = [int(r) for r in q["pool"][np.argsort(-base[q["pool"]])][:RERANK_TOP]]
            span[q["id"]] = cand
            for r in cand:
                v, n = idx.ids[r]
                ms = TMS[r]
                asr_txt = " ".join(t for x, y, t in asr_seg.get(str(v), [])
                                   if ms >= 0 and x - 5000 <= ms <= y + 5000)[:400]
                jobs.append({"key": f"{q['id']}#{r}", "query": q["text"],
                             "ocr": ocr_map.get((str(v), int(n)), ""), "asr": asr_txt})
        print(f"⑤c rerank API: {len(jobs)} khung "
              f"({len(jobs) // RERANK_BATCH + 1} lời gọi)…", flush=True)
        diem = rerank_api(jobs, a.model)
        print(f"⑤c API chấm được {len(diem)}/{len(jobs)} khung", flush=True)
        for q in Q:
            n = idx.n_frames
            sc = np.zeros(n, dtype=np.float32)
            cov = np.zeros(n, dtype=bool)
            for r in span[q["id"]]:
                d = diem.get(f"{q['id']}#{r}")
                if d is not None:
                    sc[r] = d
                    cov[r] = True
            if not cov.any():
                continue
            # Áp cho TỪNG HÀNG của S, không chỉ hàng gộp: với TRAKE, mỗi mốc có
            # bảng điểm riêng và ④ đọc cả ma trận. Chấm rerank là "khung này có
            # khớp truy vấn không" nên nó áp được cho mọi mốc như nhau; bỏ qua
            # TRAKE ở đây thì rerank chỉ có tác dụng với 21/24 đề, im lặng.
            rr = SourceScores("rerank", sc, cov)
            aw = {"fused": 1.0 / (1.0 + RERANK_W), "rerank": RERANK_W / (1.0 + RERANK_W)}
            moi = []
            for hang in q["S"]:
                nen = SourceScores("fused", hang, np.ones(n, dtype=bool))
                hop = hierarchical_rrf({"fused": [nen], "rerank": [rr]}, alpha=aw,
                                       beta={"fused": (1.0,), "rerank": (1.0,)}, k=RRF_K)
                moi.append(hop)
            q["S"] = np.vstack(moi).astype(np.float32)
        print(f"⑤c xong {time.time() - t0:.0f}s")
    else:
        print("⑤c BỎ QUA (--no-rerank)")

    # ── ④ DANTE · ⑦ ghi bài nộp theo THỂ LỆ ────────────────────────────────
    odir = Path(a.out) / "submission"
    bao = []
    for q in Q:
        n_mom = q["S"].shape[0]
        if n_mom == 1:
            order = np.argsort(-q["S"][0], kind="stable")
            lines, seen = [], set()
            for r in order[: TOP_K * 8].tolist():
                k = (str(VID[r]), int(FI[r]))
                if k in seen:
                    continue
                seen.add(k)
                lines.append((k[0], (k[1],)))
                if len(lines) >= TOP_K:
                    break
        else:
            byv = {}
            for r in q["pool"].tolist():
                byv.setdefault(str(VID[r]), []).append(r)
            sc = []
            for vid, rr in byv.items():
                seen_f, keep = set(), []
                for r in sorted(set(rr), key=lambda r: int(FI[r])):
                    f = int(FI[r])
                    if f not in seen_f:
                        seen_f.add(f)
                        keep.append(r)
                if len(keep) < n_mom:
                    continue
                cand = [int(FI[r]) for r in keep]
                for al in k_best_alignments(cand, q["S"][:, keep].tolist(),
                                            k=min(TRAKE_K_PER_VIDEO, len(keep)),
                                            pacing_penalty=a.lam):
                    sc.append((float(al.score), vid, tuple(int(f) for f in al.frames)))
            sc.sort(key=lambda x: (-x[0], x[1], x[2]))
            lines = [(v, f) for _s, v, f in sc[:TOP_K]]

        # Q&A: chưa có đường đọc ảnh ⟹ chưa sinh đáp án. Giữ chỗ để file đủ 3 cột.
        ans = "CHUA_SINH" if q["kind"] == "qa" else None
        sub = TaskSubmission(task_id=q["id"], task_type=q["kind"],
                             answers=tuple((v, f, ans) for v, f in lines),
                             n_moments=n_mom)
        p, _ = write_task_csv(sub, odir, budget=TOP_K)
        bao.append({"id": q["id"], "kind": q["kind"], "n_moc": n_mom,
                    "n_dong": len(lines), "top1": list(lines[0]) if lines else None})
        print(f"  ⑦ {q['id']:<22} {q['kind']:<6} {len(lines):>3} dòng · "
              f"top1 {lines[0][0]}@{lines[0][1][0] if lines else '-'}")

    z = pack_submission_zip(odir, Path(a.out) / "submission.zip",
                            expected=[q["id"] for q in Q])
    (Path(a.out) / "_report.json").write_text(json.dumps({
        "kien_truc": {"alpha": ALPHA, "beta": "ĐỀU 1/n trong từng modality",
                      "rrf_k": RRF_K, "pool_cap": POOL_CAP,
                      "rerank": "TẮT" if a.no_rerank else f"API {a.model}, w={RERANK_W}",
                      "lambda": a.lam, "trake_k_per_video": TRAKE_K_PER_VIDEO},
        "truy_van": bao, "giay": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✓ {len(Q)} file CSV → {z} ({z.stat().st_size / 1e3:.0f} KB) "
          f"· {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
