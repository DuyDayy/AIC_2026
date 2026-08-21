"""
Chuẩn bị THEO LÔ: chạy vòng 0 cho cả gói đề, ghi ra phiên sẵn sàng click
========================================================================

    python scripts/rf_prepare.py --dir THUNGHIEM-bo-de-thi
    python scripts/rf_server.py            # rồi mở UI, chọn phiên, click

VÌ SAO TÁCH KHỎI UI. Vòng 0 chấm BM25 trên 609.476 khung mất ~29 s mỗi truy vấn —
với 40 đề là 20 phút. Bắt operator ngồi chờ từng câu là đốt đúng thứ khan hiếm nhất
của 2h30. Chạy lô trước thì lúc mở UI mọi đề đã có sẵn top, và mỗi cú click chỉ tốn
~35 ms.

FILE PHIÊN KHÔNG PHẢI "DANH SÁCH TOP". Nó giữ `text_rrf` NGUYÊN ĐỘ DÀI 609.476 và
`q_original`, nên UI vẫn `emb @ q′` trên toàn chỉ mục sau mỗi click. Nếu chỉ lưu 100
dòng top thì Rocchio tụt xuống thành rerank danh sách cũ — đúng thứ tài liệu §2 nói
KHÔNG phải Rocchio, và khung đang nằm hạng 400.000 vĩnh viễn ở đó.

TRAKE: mỗi event thành MỘT phiên riêng, đúng §8 — *"Mỗi event Eᵢ giữ query state
riêng"*. Operator tinh chỉnh từng event như một câu KIS; DANTE ghép chuỗi ở bước
xuất, không ở bước tìm. Q&A chỉ cần đúng KHUNG ở bước này; đáp án sinh sau.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feedback.store import SessionStore                        # noqa: E402
from src.ingestion.jina_encoder import truncate_and_normalize      # noqa: E402
from src.ingestion.vector_index import load_flat_index             # noqa: E402
from src.retrieval.probe import build_probes, declarativize        # noqa: E402
from src.retrieval.score_matrix import rrf_normalize               # noqa: E402
from src.retrieval.sources import (                                # noqa: E402
    AsrSource, TextSource, VisualSource, load_asr_segments, load_frame_ms,
    load_ocr_text,
)
from src.retrieval.pool import fused_pool                          # noqa: E402
from src.submission.writer import task_type_from_filename          # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_thunghiem import nap_mo_rong, rerank_api                  # noqa: E402

RRF_K = 60.0
ALPHA = {"visual": 1 / 3, "ocr": 1 / 3, "asr": 1 / 3}
#: ⑤c hợp vào nền bằng RRF tầng ba — cùng giá trị với đường tự động.
RERANK_W, RERANK_TOP, RERANK_BATCH = 0.25, 100, 10
MODEL_SHA = "e10d47f5691d0454a0fb5d13f46f2199b74cb436"
CODE_SHA = "39e6a55ae971b59bea6e44675d237c99762e7ee2"


def tach_de(p: Path) -> list[tuple[str, str, int]]:
    """
    Một file đề → danh sách `(session_id, văn bản, event_index)`.

    TRAKE tách thành N mục: `Eᵢ` trở thành một câu KIS độc lập, đúng ý §8 — Rocchio
    chỉ lo ứng viên ngữ nghĩa của TỪNG event, còn thứ tự thời gian để DANTE lo ở
    bước xuất. KIS/QA ra đúng một mục; Q&A đi qua `declarativize` vì chỉ mục là ảnh,
    "màu áo người dẫn là gì" không giống khung nào còn "người dẫn mặc áo" thì có.
    """
    kind = task_type_from_filename(p.stem)
    raw = p.read_text(encoding="utf-8").strip()
    src = declarativize(raw) if kind == "qa" else raw
    probes = [x.text for x in build_probes(src)] or [src]
    if kind != "trake":
        return [(p.stem, probes[0] if len(probes) == 1 else src, 0)]
    return [(f"{p.stem}#E{i + 1}", t, i) for i, t in enumerate(probes)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="THUNGHIEM-bo-de-thi")
    ap.add_argument("--out", default="data/rf_sessions")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--expansions", default="expansions")
    # THỊ GIÁC GIỮ ĐÚNG MỘT RUN, và đây là lý do cơ chế chứ không phải sở thích:
    # ⑧ Rocchio chỉ cập nhật run thị giác GỐC — các bản mở rộng là câu khác, không
    # có `q₀` để cập nhật. Cho thị giác 3 run thì một cú click chỉ động vào 1/3
    # trọng số của modality đó, tức làm loãng đúng cơ chế đã chứng minh được
    # (ΔMRR +0,2557, KTC95 [+0,1680, +0,3446]). OCR/ASR không dính vì Rocchio
    # không đụng tới chúng.
    ap.add_argument("--visual-expansion", action="store_true",
                    help="thêm bản mở rộng THỊ GIÁC (mặc định TẮT — làm loãng Rocchio)")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    # ⑤c MẶC ĐỊNH TẮT. Tinh chỉnh ở đây do ⑧ Rocchio đảm nhiệm — đó là cơ chế ĐÃ
    # ĐO (ΔMRR +0,2557, KTC95 [+0,1680, +0,3446] trên 64 câu click được), còn rerank
    # bằng LLM thì CHƯA từng đo trên kho này. Chồng một cơ chế chưa đo lên một cơ chế
    # đã đo chỉ làm khó quy trách khi điểm đổi.
    ap.add_argument("--rerank", action="store_true",
                    help="bật ⑤c rerank bằng API (mặc định TẮT — dùng Rocchio)")
    a = ap.parse_args()

    de = sorted(Path(a.dir).glob("*.txt"))
    muc = [x for p in de for x in tach_de(p)]
    EXP = nap_mo_rong(Path(a.expansions))
    print(f"{len(de)} file đề → {len(muc)} phiên (TRAKE tách theo event) "
          f"· mở rộng phủ {len(EXP)} đề", flush=True)

    t0 = time.time()
    ix = load_flat_index("data/embed", dim=a.dim)
    fms = load_frame_ms()
    ocr_map = load_ocr_text("data/OCR/ocr.jsonl")
    asr_seg = load_asr_segments("data/ASR")
    tsrc = [TextSource("ocr", ix.ids, ocr_map), AsrSource(ix.ids, fms, asr_seg)]
    vis = VisualSource(ix)
    print(f"chỉ mục {time.time() - t0:.0f}s · {ix.n_frames:,} khung", flush=True)

    import torch
    from transformers import AutoModel
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True,
                                    revision=MODEL_SHA, code_revision=CODE_SHA).eval().to(dev)
    # Mã hoá câu gốc VÀ mọi bản mở rộng THỊ GIÁC — chúng là văn bản, phải qua
    # jina-clip mới thành vector để `VisualSource` chấm. (OCR/ASR thì không: chúng
    # đi qua BM25 trên chữ, không cần vector.)
    can = [t for _s, t, _e in muc]
    vis_of: dict[str, list[str]] = {}
    for sid, _t, _e in muc:
        vx = (EXP.get(sid.split("#")[0], {}).get("visual") or []) \
            if a.visual_expansion else []
        vis_of[sid] = vx
        can.extend(vx)
    with torch.inference_mode():
        V = np.asarray(enc.encode_text(can, batch_size=32), np.float32)
    VN = truncate_and_normalize(V, a.dim)
    QV = VN[: len(muc)]
    QX, off = {}, len(muc)
    for sid, _t, _e in muc:
        k = len(vis_of[sid])
        QX[sid] = VN[off:off + k]
        off += k
    del enc
    n_vx = sum(len(v) for v in vis_of.values())
    print(f"  trong đó {n_vx} bản mở rộng THỊ GIÁC", flush=True)
    print(f"① mã hoá {len(muc)} đoạn trên {dev} · {time.time() - t0:.0f}s", flush=True)

    jobs, span = [], {}
    ket: list[tuple] = []
    for i, ((sid, text, ev), qv) in enumerate(zip(muc, QV)):
        # ②③ ĐỦ 7 RUN, y hệt đường tự động.
        #
        # 🔴 Bản trước chỉ chấm câu GỐC mỗi nguồn — 3 run thay vì 7 — nên phiên UI
        # yếu sẵn từ vòng 0 so với `run_thunghiem_modal.py`, và mỗi cú click phải gỡ
        # lại phần thiếu chứ không xây thêm. [ĐO] E3: bỏ expansion mất ~0,04 MRR.
        goc = sid.split("#")[0]
        mr_all = EXP.get(goc, {})
        txt = {}
        for s in tsrc:
            mr = mr_all.get(s.name) or []
            runs = [s.score(text)] + [s.score(x) for x in mr]
            txt[s.name] = sum(
                rrf_normalize(r.scores, r.covered, RRF_K) / len(runs) for r in runs
            ).astype(np.float32)
        # Run thị giác mở rộng — TĨNH, gộp sẵn với beta chia đều cho (1 + số mở rộng)
        n_vis = 1 + len(QX[sid])
        vx = None
        if len(QX[sid]):
            vx = sum(rrf_normalize(vis.score(x).scores,
                                   np.ones(ix.n_frames, bool), RRF_K) / n_vis
                     for x in QX[sid]).astype(np.float32)
        ket.append((sid, text, ev, qv, txt, vx, n_vis))
        if a.rerank:
            hop = ALPHA["visual"] * rrf_normalize(vis.score(qv).scores,
                                                  np.ones(ix.n_frames, bool), RRF_K)
            for m, r in txt.items():
                hop = hop + ALPHA[m] * r
            pool = fused_pool(hop[None, :], RERANK_TOP)
            span[sid] = [int(r) for r in pool[np.argsort(-hop[pool])]]
            for r in span[sid]:
                v, n = ix.ids[r]
                ms = fms.get((v, int(n)), -1.0)
                asr_txt = " ".join(t for x, y, t in asr_seg.get(str(v), [])
                                   if ms >= 0 and x - 5000 <= ms <= y + 5000)[:400]
                jobs.append({"key": f"{sid}#{r}", "query": text,
                             "ocr": ocr_map.get((str(v), int(n)), ""), "asr": asr_txt})
        print(f"  ✓ {i + 1}/{len(muc)} {sid:<28} ({time.time() - t0:.0f}s)", flush=True)

    # ── ⑤c rerank bằng API, đóng băng vào phiên như một NGUỒN thứ tư ─────────
    diem = {}
    if jobs:
        print(f"\n⑤c rerank API: {len(jobs)} khung "
              f"(~{len(jobs) // RERANK_BATCH + 1} lời gọi)…", flush=True)
        diem = rerank_api(jobs, a.model)
        print(f"⑤c API chấm {len(diem)}/{len(jobs)} khung "
              f"({time.time() - t0:.0f}s)", flush=True)

    for sid, text, ev, qv, txt, vx, n_vis in ket:
        if diem and sid in span:
            sc = np.zeros(ix.n_frames, np.float32)
            cov = np.zeros(ix.n_frames, bool)
            for r in span[sid]:
                d = diem.get(f"{sid}#{r}")
                if d is not None:
                    sc[r], cov[r] = d, True
            if cov.any():
                txt = {**txt, "rerank": rrf_normalize(sc, cov, RRF_K)}
        st = SessionStore(sid.replace("#", "__"), a.out)
        them = {} if vx is None else {"vis_extra": vx}
        st.khoi_tao({"query_text": text,
                     "task": task_type_from_filename(sid.split("#")[0]),
                     "task_id": sid.split("#")[0], "event_index": ev,
                     "session_label": sid,
                     "co_rerank": "rerank" in txt, "n_visual_run": n_vis,
                     "n_run": 1 + sum(len(EXP.get(sid.split("#")[0], {}).get(m, [])) + 1
                                      for m in ("ocr", "asr"))},
                    {0: qv}, {**txt, **them})

    print(f"\n✓ {len(muc)} phiên → {a.out}/ · {time.time() - t0:.0f}s")
    print(f"  mỗi phiên: 7 run ③ + {'rerank ⑤c' if diem else 'KHÔNG rerank'}")
    print("  mở UI: python scripts/rf_server.py")


if __name__ == "__main__":
    main()
