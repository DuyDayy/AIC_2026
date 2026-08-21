"""
Công cụ tìm kiếm có phản hồi cho ngày thi — chạy TẠI MÁY, API theo tài liệu §6
==============================================================================

    python scripts/rf_server.py --port 8000     # rồi mở http://127.0.0.1:8000

VÌ SAO TẠI MÁY, KHÔNG PHẢI MODAL. Hợp đồng §6.1 đòi một vòng feedback dưới 500 ms.
Mỗi lời gọi Modal trả phí cố định ~200 s nạp chỉ mục; kể cả container ấm thì vòng
mạng cũng đã ăn hết ngân sách. Chỉ mục nằm sẵn trong RAM tại máy thì vòng feedback
là một phép nhân ma trận cục bộ.

CÁI GIÁ: khởi động mất ~90 s (đọc `emb.npy` 1,25 GB + dựng BM25 trên 609.476 khung).
Ngày thi bật server TRƯỚC khi nhận đề, không bật sau.

PHÂN VAI:
  vòng 0  — chấm đủ ba nguồn; đóng băng đóng góp RRF của OCR/ASR vào phiên
  vòng ≥1 — chỉ `emb @ q′` + hợp lại; BM25 KHÔNG chấm lại vì câu hỏi không đổi

Không dùng thư viện web ngoài: `http.server` của thư viện chuẩn là đủ cho một công
cụ một người dùng, và ngày thi thì mỗi phụ thuộc thêm là một chỗ có thể hỏng.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.feedback.rocchio import RocchioConfig, l2                 # noqa: E402
from src.feedback.session import SearchSession                     # noqa: E402
from src.feedback.store import SessionStore, liet_ke, phat_lai     # noqa: E402
from src.feedback.thumbs import KhoAnh, cat_bang_ffmpeg            # noqa: E402
from src.ingestion.jina_encoder import truncate_and_normalize      # noqa: E402
from src.ingestion.vector_index import load_flat_index             # noqa: E402
from src.retrieval.probe import build_probes, declarativize        # noqa: E402
from src.retrieval.score_matrix import rrf_normalize               # noqa: E402
from src.retrieval.sources import (                                # noqa: E402
    AsrSource, TextSource, load_asr_segments, load_frame_ms, load_ocr_text,
)
from src.submission.kbest import k_best_alignments                  # noqa: E402
from src.retrieval.pool import fused_pool                          # noqa: E402
from src.submission.writer import (                                # noqa: E402
    TaskSubmission, pack_submission_zip, task_type_from_filename, write_task_csv,
)

#: TRAKE: ④ ghép chuỗi lúc XUẤT, không lúc tìm (tài liệu §8 — Rocchio chỉ lo ứng viên
#: ngữ nghĩa của từng event, thứ tự thời gian để solver lo).
TRAKE_CAP, TRAKE_K, TRAKE_MIN_GAP, TRAKE_LAMBDA = 300, 3, 30, 0.0


def _tim_phien(so: str, kieu: str) -> list[str]:
    """`(số, kiểu)` → danh sách id phiên. TRAKE trả N phiên event theo thứ tự."""
    g = Path(ST["sess_dir"])
    if not g.is_dir():
        return []
    hop = [d.name for d in g.iterdir()
           if d.is_dir() and d.name.rsplit("__", 1)[0].endswith(f"-{so}-{kieu}")]
    return sorted(hop, key=lambda x: (len(x), x))


def xuat_trake(ids: list[str], task_id: str, out: str, top_k: int) -> dict:
    """
    Ghép N event thành chuỗi bằng ④ DANTE, rồi ghi CSV `video, f₁, …, f_N`.

    Mỗi event có bảng điểm RIÊNG (đã qua click của operator). Xếp chúng thành ma
    trận `(N, n_khung)` rồi chạy đúng `k_best_alignments` của đường tự động — không
    viết bản DP thứ hai, vì hai bản sẽ lệch.
    """
    ses = [SESS[i] if i in SESS else None for i in ids]
    if any(x is None for x in ses):
        return {"error": f"chưa mở đủ {len(ids)} phiên event: {ids}"}
    S = np.vstack([x.rank(0) for x in ses]).astype(np.float32)
    pool = fused_pool(S, TRAKE_CAP)
    byv: dict[str, list[int]] = {}
    for r in pool.tolist():
        byv.setdefault(str(ST["VID"][r]), []).append(r)
    sc = []
    for vid, rr in byv.items():
        seen, keep = set(), []
        for r in sorted(set(rr), key=lambda r: int(ST["FI"][r])):
            f = int(ST["FI"][r])
            if f not in seen:
                seen.add(f)
                keep.append(r)
        if len(keep) < S.shape[0]:
            continue
        for al in k_best_alignments([int(ST["FI"][r]) for r in keep],
                                    S[:, keep].tolist(),
                                    k=min(TRAKE_K, len(keep)),
                                    min_gap=TRAKE_MIN_GAP,
                                    pacing_penalty=TRAKE_LAMBDA):
            sc.append((float(al.score), vid, tuple(int(f) for f in al.frames)))
    sc.sort(key=lambda x: (-x[0], x[1], x[2]))
    lines, seen2 = [], set()
    for _s, v, f in sc:
        if (v, f) in seen2:
            continue
        seen2.add((v, f))
        lines.append((v, f))
        if len(lines) >= top_k:
            break
    if not lines:
        return {"error": "không video nào đủ N khung trong rổ — nới TRAKE_CAP"}
    sub = TaskSubmission(task_id, "trake",
                         tuple((v, f, None) for v, f in lines),
                         n_moments=S.shape[0])
    pth, _ = write_task_csv(sub, Path(out) / "submission", budget=top_k)
    return {"file": str(pth), "n_dong": len(lines), "n_event": S.shape[0],
            "dong_1": [lines[0][0], list(lines[0][1])]}

ALPHA = {"visual": 1 / 3, "ocr": 1 / 3, "asr": 1 / 3}
RRF_K = 60.0
#: Số khung HIỆN cho operator. Rổ đúng 300 vì [ĐO] trung vị hạng của khung đúng ở
#: những câu rớt top-100 là **381** — hiện 40 thì phần lớn câu không có gì để bấm,
#: và đó là giới hạn của GIAO DIỆN, không phải của Rocchio.
#: Hiện 300 ảnh chỉ khả thi từ khi đọc kho keyframe: [ĐO] 4–7 ms cho 40 ảnh, còn
#: ffmpeg không `-ss` thì 6,92 s MỖI ảnh — 300 ảnh sẽ là nửa giờ.
XEM_K = 300
#: Bài nộp vẫn đúng 100 dòng theo thể lệ, bất kể xem bao nhiêu.
NOP_K = 100
#: ⑤c hợp vào nền bằng RRF TẦNG BA, cùng cách với đường tự động: nền giữ
#: `1/(1+w)`, rerank lấy `w/(1+w)`. Phải chuẩn hoá lại, nếu không thêm một thành
#: phần là âm thầm tăng tổng trọng số và đổi cả cân bằng ba nguồn kia.
RERANK_W = 0.25


def alpha_cho(thanh_phan) -> dict[str, float]:
    """`alpha` khớp ĐÚNG các thành phần phiên đang có."""
    if "rerank" not in thanh_phan:
        return dict(ALPHA)
    z = 1.0 + RERANK_W
    return {**{k: v / z for k, v in ALPHA.items()}, "rerank": RERANK_W / z}

ST: dict = {}
SESS: dict[str, SearchSession] = {}
LOCK = threading.Lock()


def nap(dim: int, model_sha: str, code_sha: str) -> None:
    t0 = time.time()
    ix = load_flat_index("data/embed", dim=dim)
    fms = load_frame_ms()
    ocr = load_ocr_text("data/OCR/ocr.jsonl")
    asr = load_asr_segments("data/ASR")
    ST.update(
        ix=ix, ids=list(ix.ids), FI=np.asarray(ix.frame_idx), ocr=ocr, asr=asr,
        VID=np.array([v for v, _ in ix.ids]),
        TS=np.array([fms.get((v, int(n)), -1.0) for v, n in ix.ids], np.float64) / 1000.0,
        tsrc=[TextSource("ocr", ix.ids, ocr), AsrSource(ix.ids, fms, asr)],
        dim=dim, model_sha=model_sha, code_sha=code_sha, enc=None,
    )
    t1 = time.time()
    kho = KhoAnh()
    kho.mo_san()
    ST["kho"] = kho
    print(f"✓ kho ảnh {len(kho):,} khung · {time.time() - t1:.1f}s "
          f"(khớp chỉ mục: {len(kho) == ix.n_frames})", flush=True)
    print(f"✓ chỉ mục sẵn sàng {time.time() - t0:.0f}s · {ix.n_frames:,} khung "
          f"· {len(ix.ranges)} video", flush=True)


def encode(text: str) -> np.ndarray:
    """Mã hoá tại máy, GHIM revision — vector khung dùng đúng cặp sha này."""
    if ST["enc"] is None:
        import torch
        from transformers import AutoModel
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        ST["enc"] = AutoModel.from_pretrained(
            "jinaai/jina-clip-v2", trust_remote_code=True,
            revision=ST["model_sha"], code_revision=ST["code_sha"]).eval().to(dev)
        print(f"✓ encoder trên {dev}", flush=True)
    import torch
    with torch.inference_mode():
        v = np.asarray(ST["enc"].encode_text([text], batch_size=1), np.float32)
    return truncate_and_normalize(v, ST["dim"])[0]


def anh_nhieu(rows: list[int]) -> list[str]:
    """
    Ảnh của nhiều khung → base64. Kho keyframe trước, ffmpeg là ĐƯỜNG LÙI.

    [ĐO] kho: 4–7 ms cho 40 ảnh. ffmpeg `-ss`: ~70 ms/ảnh. ffmpeg không `-ss`
    (bản đầu của tôi): 6,92 s/ảnh, tức 40 ảnh mất 2–4 phút và UI trông như treo.
    """
    keys = [(str(ST["VID"][r]), int(ST["ids"][r][1])) for r in rows]
    out = ST["kho"].doc_nhieu(keys)
    for i, (b64, r) in enumerate(zip(out, rows)):
        if not b64:
            b = cat_bang_ffmpeg(str(ST["VID"][r]), float(ST["TS"][r]))
            out[i] = base64.b64encode(b).decode() if b else ""
    return out


def khung_jpeg(row: int) -> str:
    """Một khung — dùng cho ⑥ Qwen. Kho trả WebP; Qwen nhận được cả hai."""
    return anh_nhieu([row])[0]


def ket_qua(s: SearchSession, event_id: int, k: int, anh: bool) -> list[dict]:
    sc = s.rank(event_id)
    order = np.argsort(-sc, kind="stable")[: k * 4]
    out, seen = [], set()
    for r in order.tolist():
        key = (str(ST["VID"][r]), int(ST["FI"][r]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"row": r, "video_id": key[0], "frame_id": key[1],
                    "t": round(float(ST["TS"][r]), 2), "score": float(sc[r]),
                    "img": ""})
        if len(out) >= k:
            break
    if anh and out:
        # dựng ảnh theo LÔ: một lời gọi song song thay vì `k` lời gọi nối đuôi
        for x, b64 in zip(out, anh_nhieu([x["row"] for x in out])):
            x["img"] = b64
    return out


def moi_phien(query: str, task: str, cfg: RocchioConfig, k: int, anh: bool) -> dict:
    """Vòng 0 — chấm đủ ba nguồn, rồi ĐÓNG BĂNG phần văn bản vào phiên."""
    t0 = time.time()
    src = declarativize(query) if task == "qa" else query
    probes = [p.text for p in build_probes(src)] or [src]
    n = ST["ix"].n_frames
    text_rrf = {}
    for s in ST["tsrc"]:
        acc = np.zeros(n, np.float32)
        for t in probes:
            x = s.score(t)
            acc += rrf_normalize(x.scores, x.covered, RRF_K) / len(probes)
        text_rrf[s.name] = acc
    ses = SearchSession(ST["ix"], text_rrf, ST["TS"], ST["VID"],
                        alpha_cho(text_rrf), rrf_k=RRF_K, cfg=cfg, task=task,
                        query_text=query)
    for i, t in enumerate(probes):
        ses.khoi_tao(encode(t), event_id=i)
    st = SessionStore(ses.session_id, ST["sess_dir"])
    st.khoi_tao({"query_text": query, "task": task, "task_id": "", "event_index": 0,
                 "session_label": ses.session_id}, {i: ses.state(i).q_original
                                                    for i in range(len(probes))},
                text_rrf)
    ses.store = st
    SESS[ses.session_id] = ses
    return {"session_id": ses.session_id, "round": 0, "n_events": len(probes),
            "vong0_ms": round((time.time() - t0) * 1000),
            "results": ket_qua(ses, 0, k, anh)}


def sinh_dap_an_qwen(ses: SearchSession, row: int) -> dict:
    """
    ⑥ — gọi Qwen2.5-VL trên Modal đọc ĐÚNG khung operator đang chọn.

    Ảnh cắt từ VIDEO GỐC theo `frame_idx`. Không lấy từ kho keyframe: kho đó đánh số
    theo `n` của bộ trích xuất CŨ (601 ảnh cho `L21_V001` trong khi chỉ mục nay có
    1.778 khung), nên tra ở đó ra ẢNH KHÁC mà không gì báo.
    """
    import modal

    b64 = khung_jpeg(row)
    if not b64:
        return {"error": f"không cắt được khung {row} — thiếu .mp4?"}
    vid, n = ST["ids"][row]
    ms = ST["TS"][row] * 1000.0
    asr = " ".join(t for x, y, t in ST["asr"].get(str(vid), [])
                   if ms >= 0 and x - 5000 <= ms <= y + 5000)[:400]
    fn = modal.Function.from_name("aic-query", "qa_answer")
    got = fn.remote([{"image_b64": b64, "question": ses.query_text,
                      "ocr": ST["ocr"].get((str(vid), int(n)), ""), "asr": asr}])
    return {"answer": ses.dat_answer(got[0] if got else ""),
            "khung": f"{vid}@{int(ST['FI'][row])}"}


def xuat(s: SearchSession, d: dict) -> dict:
    """
    Ghi `<out>/submission/<task_id>.csv` theo THỂ LỆ, qua đúng `write_task_csv`.

    Dùng lại hàm của đường chạy thi chứ không tự ghi CSV: cổng Tầng 0 (bù δ), phép
    bọc ngoặc kép cột đáp án, phép đọc-ngược-tự-kiểm và trần 100 dòng đều nằm ở đó.
    Tự nối chuỗi ở đây là dựng bản sao thứ hai của định dạng, và bản sao sẽ lệch.
    """
    task_id = str(d["task_id"])
    kind = task_type_from_filename(task_id)
    lines = s.dong_nop(int(d.get("top_k", NOP_K)), int(d.get("event_id", 0)))
    if kind == "qa" and not (s.answer or "").strip():
        return {"error": "đề Q&A chưa có đáp án — POST /answer trước, "
                         "nếu không câu này chắc chắn 0 điểm"}
    sub = TaskSubmission(task_id=task_id, task_type=kind,
                         answers=tuple((v, f, s.answer if kind == "qa" else None)
                                       for v, f in lines),
                         n_moments=len(lines[0][1]) if lines else 1)
    out = Path(d.get("out", "submission_rf")) / "submission"
    pth, notes = write_task_csv(sub, out, budget=int(d.get("top_k", NOP_K)))
    return {"file": str(pth), "n_dong": len(lines), "n_chot": len(s.chot),
            "answer": s.answer, "ghi_chu": notes,
            "dong_1": [lines[0][0], list(lines[0][1])] if lines else None}


def mo_phien(sid: str, beta: float, k: int, anh: bool,
             gam: float = 0.0) -> dict:
    """
    Nạp một phiên đã chuẩn bị theo lô. KHÔNG chấm lại BM25 — nó nằm sẵn trong
    `vectors.npz`, nên mở phiên tốn mili-giây thay vì 29 giây.

    Nhật ký được PHÁT LẠI trước khi gắn `store`, để lần khôi phục không tự ghi
    thêm bản sao của chính nó vào nhật ký.
    """
    st = SessionStore(sid, ST["sess_dir"])
    meta, q0, txt, ev = st.doc()
    vx = txt.pop("vis_extra", None)
    ses = SearchSession(ST["ix"], txt, ST["TS"], ST["VID"], alpha_cho(txt),
                        visual_extra=vx,
                        n_visual_run=int(meta.get("n_visual_run", 1)),
                        rrf_k=RRF_K,
                        cfg=RocchioConfig(beta=beta, gamma=gam), task=meta.get("task", "kis"),
                        query_text=meta.get("query_text", ""))
    ses.session_id = sid
    for e, v in q0.items():
        ses.khoi_tao(np.asarray(v, np.float32), event_id=int(e))
    phat_lai(ses, ev)
    ses.store = st                       # gắn SAU khi phát lại
    SESS[sid] = ses
    return {"session_id": sid, "round": ses.round, "meta": meta,
            "truy_van": meta.get("query_text", ""),
            "chot": [[v, list(f)] for v, f in ses.chot],
            "n_events_log": len(ev), "n_chot": len(ses.chot), "answer": ses.answer,
            "results": ket_qua(ses, 0, k, anh)}


UI = """<!doctype html><meta charset=utf-8><title>RF · AIC 2026</title>
<style>
body{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{position:sticky;top:0;background:#1b1b1b;padding:10px 14px;border-bottom:1px solid #333;
 display:flex;gap:8px;align-items:center;flex-wrap:wrap;z-index:9}
input,select,button{font:inherit;padding:6px 10px;border-radius:6px;border:1px solid #444;
 background:#222;color:#eee}button{cursor:pointer}
button.go{background:#1d4ed8;border-color:#1d4ed8}button.ex{background:#166534;border-color:#166534}
#q{line-height:1.45}#bar{padding:6px 14px;color:#9ca3af;font-size:13px;border-bottom:1px solid #333}
#tabs{padding:6px 14px;display:flex;gap:6px;flex-wrap:wrap}
#tabs button{padding:4px 12px;font-size:13px}#tabs button.on{background:#1d4ed8;border-color:#1d4ed8}
#g{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;padding:14px}
.c{background:#1b1b1b;border:1px solid #333;border-radius:8px;overflow:hidden}
.c img{width:100%;display:block;background:#000;aspect-ratio:16/9;object-fit:contain}
.m{padding:6px 8px;font-size:12px;color:#9ca3af}.a{display:flex;gap:4px;padding:0 8px 8px}
.a button{flex:1;padding:4px}.sel{outline:2px solid #16a34a}
</style>
<header>
 <b>Câu</b><input id=so type=number value=1 style=width:70px>
 <select id=kieu><option>kis<option>qa<option>trake</select>
 <button class=go onclick=mo()>Mở</button>
 <label>β <input id=beta type=number step=.05 value=.8 style=width:66px></label>
 <button onclick=act('undo')>Undo</button><button onclick=act('reset')>Reset</button>
 <span style=flex:1></span>
 <input id=ans placeholder="đáp án Q&A" style=width:190px>
 <button onclick=qwen()>Qwen đọc khung</button>
 <button class=ex onclick=xuat()>Xuất CSV</button>
 <button class=ex onclick=goi()>Đóng gói .zip</button>
 <button style=background:#7f1d1d;border-color:#7f1d1d onclick=tat()>⏻ Tắt</button>
</header>
<div id=q style="padding:8px 14px;background:#16213a;border-bottom:1px solid #333;
 font-size:14px;color:#dbeafe">— chưa mở phiên —</div>
<div id=bar>nhập số câu rồi bấm Mở — hiện 300 khung, nộp 100 dòng</div><div id=tabs></div><div id=g></div>
<script>
const XK=300, NK=100;
let S=null,IDS=[],EV=0,CH=[],LAST=null,BAN=false,QTXT='';
function hienq(){q.textContent=(IDS.length>1?('E'+(EV+1)+'/'+IDS.length+' · '):'')
 +(QTXT||'—');}
// Mọi lỗi PHẢI hiện ra. `fetch` KHÔNG ném ở 4xx/5xx — nó trả về bình thường, nên
// bản trước gọi thẳng `r.cos.toFixed(3)` và chết im khi server trả {error}.
// Đó chính là lý do bấm nút mà không thấy gì xảy ra.
async function J(u,d){
 if(BAN){bar.textContent='⏳ đang bận, chờ lượt trước xong…';return null;}
 BAN=true;const cu=bar.textContent;bar.textContent='⏳ '+u.slice(1)+'…';
 try{
  const res=await fetch(u,{method:'POST',body:JSON.stringify(d)});
  const j=await res.json().catch(()=>({error:'server trả về không phải JSON'}));
  if(!res.ok||j.error){bar.textContent='✗ '+(j.error||('HTTP '+res.status));return null;}
  return j;
 }catch(e){bar.textContent='✗ '+e.message+' — server còn chạy không?';return null;}
 finally{BAN=false;if(bar.textContent.startsWith('⏳'))bar.textContent=cu;}}
const tid=()=>'query-p1-'+so.value+'-'+kieu.value;
const canS=()=>{if(!S){bar.textContent='✗ chưa mở phiên nào — bấm Mở trước';return false}return true};

async function mo(){
 const r=await J('/mo',{so:so.value,kieu:kieu.value,beta:+beta.value,
  k:XK});
 if(!r)return;
 IDS=r.ids;EV=0;S=IDS[0];CH=r.chot||[];ans.value=r.answer||'';
 QTXT=r.truy_van||'';hienq();
 tab();draw(r,'mở '+tid()+' · '+r.n_event+' event · vòng '+r.round);}

function tab(){tabs.innerHTML='';if(IDS.length<2)return;
 IDS.forEach((id,i)=>{const b=document.createElement('button');
  b.textContent='E'+(i+1);if(i==EV)b.className='on';
  b.onclick=()=>doi(i);tabs.appendChild(b);});}

async function doi(i){EV=i;S=IDS[i];tab();
 const r=await J('/open',{session_id:S,beta:+beta.value,k:XK});
 if(!r)return;QTXT=r.truy_van||'';hienq();draw(r,'E'+(i+1)+' · vòng '+r.round);}

async function fb(row,pos){
 if(!canS())return;
 if(!pos){bar.textContent='👎 đã ghi nhật ký nhưng CHƯA đổi thứ hạng: γ=0. '
  +'Tài liệu §14 để negative feedback sau feature flag, mặc định TẮT.';}
 const r=await J('/feedback',{session_id:S,row:row,positive:pos,k:XK});
 if(!r)return;
 draw(r,(pos?'👍':'👎 (γ=0, không đổi thứ hạng)')+' vòng '+r.round+' · '
  +r.latency_ms+' ms · cos='+r.cos.toFixed(3)+' · β='+r.beta.toFixed(2)
  +' · γ='+(r.gamma||0)
  +(r.drift?' · ⚠TRÔI':''));}

async function chot(row){if(!canS())return;
 const r=await J('/submit',{session_id:S,row:row,k:XK});
 if(!r)return;CH=r.chot;draw(r,'✓ đã chốt '+r.n_chot+' khung (ghim lên đầu bài nộp)');}

async function act(a){if(!canS())return;const r=await J('/'+a,{session_id:S,k:XK});
 if(r)draw(r,a+' · vòng '+r.round);}

async function qwen(){if(!canS())return;
 if(LAST===null){bar.textContent='✗ bấm vào một ảnh để chọn khung trước';return;}
 const r=await J('/qwen',{session_id:S,row:LAST});
 if(!r)return;ans.value=r.answer;bar.textContent='Qwen · '+r.khung+' → '+r.answer;}

async function xuat(){if(!canS())return;
 if(ans.value)await J('/answer',{session_id:S,answer:ans.value});
 const r=IDS.length>1
  ? await J('/export_trake',{ids:IDS,task_id:tid(),out:'submission_rf'})
  : await J('/export',{session_id:S,task_id:tid(),out:'submission_rf'});
 if(!r)return;
 bar.textContent='✓ '+r.file+' · '+r.n_dong+' dòng · dòng1 '+JSON.stringify(r.dong_1);}

async function tat(){
 if(!confirm('Tắt server? Mọi thao tác đã lưu xuống đĩa, bật lại là khôi phục.'))return;
 try{const r=await J('/tat',{});if(r)bar.textContent='⏻ '+r.msg;}catch(e){bar.textContent='⏻ đã tắt';}
 document.body.style.opacity=.4;}
async function goi(){const r=await J('/package',{out:'submission_rf'});
 if(r)bar.textContent='✓ '+r.zip+' · '+r.kb+' KB';}

function draw(r,msg){
 bar.textContent=msg+' · hiện '+((r.results||[]).length)+' khung, sẽ nộp '+NK+' dòng';
 g.innerHTML='';
 if(r.results&&r.results.length)LAST=r.results[0].row;
 (r.results||[]).forEach((x,i)=>{const d=document.createElement('div');d.className='c';
  d.innerHTML='<img src="data:image/webp;base64,'+x.img+'">'
   +'<div class=m>#'+(i+1)+' · '+x.video_id+' · '+x.frame_id+' · '+x.t+'s</div>'
   +'<div class=a><button title="thêm vào D⁺ — đổi truy vấn, tìm lại toàn chỉ mục">👍</button>'
   +'<button title="ghi vào D⁻ — γ=0 nên CHƯA đổi thứ hạng" style=opacity:.45>👎</button>'
   +'<button title="ghim khung này lên dòng 1 bài nộp">✓</button></div>';
  const bs=d.querySelectorAll('.a button');
  bs[0].onclick=()=>{LAST=x.row;fb(x.row,true)};
  bs[1].onclick=()=>fb(x.row,false);
  bs[2].onclick=()=>{LAST=x.row;chot(x.row)};
  d.querySelector('img').onclick=()=>{LAST=x.row;bar.textContent='chọn '+x.video_id+'@'+x.frame_id};
  if(CH.some(c=>c[0]==x.video_id&&c[1][0]==x.frame_id))d.classList.add('sel');
  g.appendChild(d);});}
</script>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if urlparse(self.path).path != "/":
            return self._send({"error": "not found"}, 404)
        b = UI.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            d = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send({"error": f"json hỏng: {e}"}, 400)
        p = urlparse(self.path).path
        k, ev, anh = int(d.get("k", XEM_K)), int(d.get("event_id", 0)), d.get("img", True)
        try:
            with LOCK:
                if p == "/sessions":
                    return self._send({"sessions": liet_ke(ST["sess_dir"])})
                if p == "/mo":
                    ids = _tim_phien(str(d["so"]), str(d["kieu"]))
                    if not ids:
                        return self._send({"error": f"không có phiên "
                                                    f"{d['so']}-{d['kieu']}"}, 400)
                    gam = float(d.get("gamma", 0.0))
                    r = [mo_phien(i, float(d.get("beta", 0.8)), k,
                                  anh and i == ids[0], gam) for i in ids]
                    return self._send({"ids": ids, "n_event": len(ids), **r[0]})
                if p == "/qwen":
                    return self._send(sinh_dap_an_qwen(SESS[d["session_id"]],
                                                       int(d["row"])))
                if p == "/export_trake":
                    return self._send(xuat_trake(list(d["ids"]), str(d["task_id"]),
                                                 d.get("out", "submission_rf"),
                                                 int(d.get("top_k", NOP_K))))
                if p == "/open":
                    return self._send(mo_phien(str(d["session_id"]),
                                               float(d.get("beta", 0.8)), k, anh,
                                               float(d.get("gamma", 0.0))))
                if p == "/search":
                    cfg = RocchioConfig(beta=float(d.get("beta", 0.8)))
                    return self._send(moi_phien(d["query"], d.get("task", "kis"),
                                                cfg, k, anh))
                # ── Các đường KHÔNG cần phiên phải đứng TRƯỚC dòng tra `SESS` ──
                # 🔴 Lỗi đã xảy ra: `/package` và `/tat` nằm SAU nó, nên bấm "Đóng gói
                # .zip" báo `thiếu/không có: 'session_id'` dù thao tác đó chỉ nén thư
                # mục, không liên quan phiên nào. Thứ tự định tuyến LÀ hợp đồng API.
                if p == "/package":
                    out = Path(d.get("out", "submission_rf"))
                    z = pack_submission_zip(out / "submission", out / "submission.zip")
                    return self._send({"zip": str(z),
                                       "kb": round(z.stat().st_size / 1e3)})
                if p == "/tat":
                    # Trả lời XONG rồi mới tắt, để trình duyệt nhận được xác nhận.
                    threading.Timer(0.5, lambda: os._exit(0)).start()
                    return self._send({"ok": True,
                                       "msg": "server đang tắt — nhật ký đã lưu"})

                if "session_id" not in d or d["session_id"] is None:
                    return self._send({"error": f"{p} cần session_id — "
                                                f"bấm Mở phiên trước"}, 400)
                if d["session_id"] not in SESS:
                    return self._send({"error": f"phiên {d['session_id']!r} chưa mở "
                                                f"trong server này — bấm Mở lại"}, 400)
                s = SESS[d["session_id"]]
                if p == "/feedback":
                    r = s.feedback(int(d["row"]), bool(d.get("positive", True)), ev)
                    r["truy_van"] = s.query_text
                    r["gamma"] = s.cfg.gamma
                    r["results"] = ket_qua(s, ev, k, anh)
                    return self._send(r)
                if p == "/submit":
                    n = s.submit(int(d["row"]))
                    return self._send({"round": s.round, "n_chot": n,
                                       "chot": [[v, list(f)] for v, f in s.chot],
                                       "results": ket_qua(s, ev, k, anh)})
                if p == "/answer":
                    return self._send({"answer": s.dat_answer(d.get("answer", ""))})
                if p == "/export":
                    return self._send(xuat(s, d))
                if p == "/undo":
                    s.undo()
                elif p == "/reset":
                    s.reset()
                else:
                    return self._send({"error": "not found"}, 404)
                return self._send({"round": s.round, "results": ket_qua(s, ev, k, anh)})
        except KeyError as e:
            return self._send({"error": f"thiếu/không có: {e}"}, 400)
        except Exception as e:                                     # noqa: BLE001
            return self._send({"error": f"{type(e).__name__}: {e}"}, 500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--model-sha", default="e10d47f5691d0454a0fb5d13f46f2199b74cb436")
    ap.add_argument("--code-sha", default="39e6a55ae971b59bea6e44675d237c99762e7ee2")
    ap.add_argument("--sessions", default="data/rf_sessions")
    a = ap.parse_args()
    print("nạp chỉ mục (~90 s) — ngày thi hãy bật TRƯỚC khi nhận đề…", flush=True)
    nap(a.dim, a.model_sha, a.code_sha)
    ST["sess_dir"] = a.sessions
    ds = liet_ke(a.sessions)
    print(f"✓ {len(ds)} phiên đã chuẩn bị sẵn trong {a.sessions}/", flush=True)
    print(f"→ http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
