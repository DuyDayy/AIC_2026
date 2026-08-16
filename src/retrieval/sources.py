"""
Tầng ② — BỐN NGUỒN ĐIỂM: mỗi nguồn trả lời một câu hỏi khác nhau
=================================================================

    thị giác  jina-clip-v2      "khung này TRÔNG giống mô tả không?"
    ASR       BM25 trên lời nói "có ai NÓI câu đó không?"
    OCR       BM25 trên chữ     "có CHỮ đó trên màn hình không?"
    object    BM25 trên bố cục  "có VẬT đó, ở đúng chỗ, đúng số lượng không?"

Mọi nguồn trả về `SourceScores` — điểm cho **từng khung của chỉ mục**, cùng thứ tự với
`FlatIndex.ids`, kèm mặt nạ `covered` nói khung nào nguồn đó thật sự có dữ liệu.

=============================================================================
VÌ SAO PHẢI CÓ `covered`, KHÔNG CHỈ CÓ ĐIỂM
=============================================================================

Điểm 0 có hai nghĩa hoàn toàn khác nhau: **"có dữ liệu và không khớp"** so với
**"không có dữ liệu"**. Gộp chúng làm một là đúng cái lỗi đã giết RRF: nguồn phủ một
phần được cộng điểm dương *bất kể liên quan*, làm 216 video được nâng hạng có hệ thống
và mất **−0,2927 FINAL** trên 688 truy vấn.

Tầng ③ cần phân biệt hai nghĩa đó để chuẩn hoá z **chỉ trên tập có dữ liệu**. Nên
`covered` không phải tiện ích — nó là điều kiện để ③ đúng.

=============================================================================
PHẠM VI KHÁC NHAU: ASR THEO THỜI GIAN, BA NGUỒN KIA THEO KHUNG
=============================================================================

ASR là các đoạn `(start_ms, end_ms, text)`. Một khung không "có" transcript — nó **rơi
vào** một khoảng thời gian. Nên `AsrSource` gán cho mỗi khung phần lời nói trong một
CỬA SỔ quanh mốc thời gian của nó.

Cửa sổ mặc định ±5 giây, và đó là **lựa chọn có đánh đổi**: hẹp quá thì khung im lặng
không có gì, rộng quá thì mọi khung trong một video dài đều mang gần cùng một đoạn text
và ASR mất khả năng phân biệt *trong* video. Ghi vào `window_ms` để đổi và đo được.

=============================================================================
ĐỘ PHỦ THẬT, ĐO 2026-08-14 — không phải con số trong tài liệu cũ
=============================================================================

    thị giác   173.426/173.426 = 100%
    OCR        169.409/173.426 =  97,7%   ← không còn là nguồn phủ một phần
    object     173.426/173.426 = 100%     (nhưng 80% chỉ hợp lệ ở mức CẢNH)
    ASR        873/873 video               (theo cửa sổ thời gian, xem trên)

OCR từng là 216/873 video = 24,7%. Nó đã được chạy lại đủ. Mọi câu trong tài liệu cũ
dựa trên 24,7% cần đọc lại — gồm cả lý do RRF hỏng, tuy kết luận không đổi vì phép
chuẩn hoá z đúng bất kể độ phủ.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from src.text.fold import strip_diacritics
from src.retrieval.bm25 import DEFAULT_B, DEFAULT_K1, Bm25Index

FrameKey = tuple[str, int]

# Cửa sổ thời gian gán lời nói cho một khung. Xem docstring.
# =============================================================================
# ĐÃ QUÉT — VÀ CỬA SỔ HOÁ RA GẦN NHƯ KHÔNG PHẢI THAM SỐ
# =============================================================================
#
# [ĐO] 226 truy vấn, thị giác 512 chiều + ASR, δ = 0,10:
#
#     cửa sổ   độ phủ   R@10    R@100   Final    thắng/thua vs KHÔNG ASR   p
#     0s          93%   0,420   0,681   0,4451                    73/95    0,105
#     2s          96%   0,389   0,681   0,4434                    79/93    0,322
#     5s          97%   0,385   0,686   0,4487                    72/97    0,065  ← giữ
#     15s         98%   0,394   0,681   0,4451                    75/86    0,431
#     45s         98%   0,389   0,699   0,4522                    80/85    0,756
#     chỉ thị giác  —   0,407   0,690   0,4566
#
# HAI ĐIỀU BẤT NGỜ.
#
# 1. **Độ phủ đã 93% ngay ở cửa sổ 0.** Đoạn Whisper dài hàng chục giây nên gần như
#    mọi keyframe vốn đã nằm TRONG một đoạn. Nới cửa sổ từ 0 lên 45 giây chỉ thêm 5
#    điểm phần trăm phủ. Đánh đổi "rộng thì mất khả năng phân biệt, hẹp thì khung im
#    lặng trống" mà docstring dưới mô tả **gần như không tồn tại ở kho này**.
#
# 2. **Không cửa sổ nào thắng nổi chỉ-thị-giác** trên bộ eval này. Nhưng đó là điều
#    ĐÃ BIẾT TRƯỚC: 226 câu do model nhìn ảnh viết ra nên thuần thị giác, lời nói
#    không thể giúp. Con số này KHÔNG nói ASR vô dụng — nó nói bộ eval mù với ASR.
#
# Giữ 5.000 ms: nằm giữa dải, và mọi giá trị trong dải đều không phân biệt được. Đo
# lại khi có truy vấn thật sự cần lời nói.
ASR_WINDOW_MS = 5_000


# =============================================================================
# BM25 `k1`, `b` — RIÊNG TỪNG NGUỒN, KHÔNG DÙNG CHUNG
# =============================================================================
#
# `b` là mức chuẩn hoá theo ĐỘ DÀI văn bản. Ba nguồn muốn `b` **ngược nhau**, nên một bộ
# tham số chung là sai thiết kế.
#
# [ĐO] MRR của khung đáp án, bộ giữ kín 110 câu, quét 36 tổ hợp:
#
#     b tại k1=1,5      OCR       ASR
#     0,0            0,0194    0,1816   ← ASR đỉnh
#     0,5            0,1625    0,1616
#     0,75           0,1660    0,1505   ← mặc định cũ, không tối ưu cho nguồn nào
#     0,9            0,1641    0,1520
#
#     tối ưu riêng    k1     b      MRR    so với (1,5/0,75)
#     OCR            0,6   0,9   0,1730         +0,0070
#     ASR            1,5   0,0   0,1816         +0,0311
#     vật thể        2,5   0,0   0,0003         +0,0001
#
# **Cơ chế**, không phải trùng hợp:
#   · OCR là chữ chạy trên màn, độ dài RẤT chênh — khung nhiều chữ dễ khớp bừa, nên cần
#     chuẩn hoá mạnh (`b ≈ 0,9`).
#   · ASR lấy trong cửa sổ ±4 giây CỐ ĐỊNH nên độ dài gần như đều; chuẩn hoá theo độ dài
#     ở đây chỉ thêm nhiễu (`b = 0`).
#
# ⚠️ Nguồn vật thể có MRR **0,0003** — hạng đáp án cỡ 3.000. Chỉnh `k1`/`b` cho nó là vô
# nghĩa; vấn đề nằm ở chất lượng nhãn, không ở tham số.
#
# [ĐO ĐẦU-CUỐI] và đây là chỗ phải thành thật: lợi ở mức nguồn **KHÔNG** chuyển thành
# `Final` một cách chắc chắn. Bộ giữ kín, cùng rải thích ứng, cùng không VLM:
#
#     L      k1/b chung   riêng nguồn        Δ                KTC95    T/Th
#     9         0,2456        0,2501   +0,0045   [−0,0036,+0,0135]    10/6  —
#     11        0,2678        0,2727   +0,0049   [−0,0045,+0,0145]    10/6  —
#     21        0,2833        0,2901   +0,0068   [−0,0039,+0,0182]    10/6  —
#
# Dương ở mọi `L` nhưng **KTC chứa 0**; chỉ 16/110 câu đổi. GIỮ thay đổi vì lý do CƠ CHẾ
# chứ không vì con số: `b = 0` cho ASR suy ra từ **thiết kế nguồn** (cửa sổ ±5 giây cố
# định ⟹ độ dài đều ⟹ chuẩn hoá theo độ dài là nhiễu thuần), nên nó đúng bất kể bộ eval.
# Đó là khác biệt giữa cái này và một tham số chọn bằng argmax.
BM25_PARAMS: dict[str, tuple[float, float]] = {
    "ocr": (0.6, 0.9),
    "asr": (1.5, 0.0),
    "object": (DEFAULT_K1, DEFAULT_B),      # giữ mặc định: chỉnh cũng vô nghĩa
}

@dataclass(frozen=True)
class SourceScores:
    """
    Điểm của MỘT nguồn cho toàn bộ khung của chỉ mục.

    `scores[i]` và `covered[i]` ứng với `FlatIndex.ids[i]` — hai mảng song song, và
    lệch nhau là lỗi im lặng, nên `__post_init__` kiểm.
    """

    name: str
    scores: np.ndarray      # (N,) float32
    covered: np.ndarray     # (N,) bool — nguồn này CÓ dữ liệu cho khung đó

    def __post_init__(self) -> None:
        if self.scores.shape != self.covered.shape:
            raise ValueError(
                f"{self.name}: điểm {self.scores.shape} ≠ mặt nạ {self.covered.shape} — "
                f"hai mảng song song đã lệch"
            )

    @property
    def coverage(self) -> float:
        return float(self.covered.mean()) if self.covered.size else 0.0


class VisualSource:
    """Cosine với chỉ mục vector. Phủ 100% theo định nghĩa — mọi khung đều có vector."""

    name = "visual"

    def __init__(self, index) -> None:
        self.index = index

    def score(self, query_vec: np.ndarray) -> SourceScores:
        q = np.asarray(query_vec, dtype=np.float32).ravel()
        if q.shape[0] != self.index.dim:
            raise ValueError(f"truy vấn {q.shape[0]} chiều ≠ chỉ mục {self.index.dim}")
        s = (self.index.emb @ q).astype(np.float32)
        return SourceScores(self.name, s, np.ones(s.shape[0], dtype=bool))


class TextSource:
    """
    Nguồn văn bản theo khung, chấm bằng BM25.

    Dùng chung cho OCR và object: cả hai đều là `{khung → chuỗi}`. Khác nhau chỉ ở chỗ
    lấy chuỗi từ đâu, nên `from_map` nhận sẵn dict thay vì tự đọc file — nhờ vậy test
    được mà không cần dữ liệu thật.
    """

    def __init__(self, name: str, ids: Sequence[FrameKey],
                 text_of: Mapping[FrameKey, str],
                 k1: float | None = None, b: float | None = None) -> None:
        """`k1`/`b` là tham số BM25. `None` ⟹ lấy theo tên nguồn từ `BM25_PARAMS`."""
        self.name = name
        self._n = len(ids)
        # `doc_id` của BM25 là CHUỖI và phải duy nhất; dùng chỉ số hàng để ánh xạ
        # ngược không mơ hồ. Chỉ đưa vào khung CÓ chữ — khung rỗng không phải "không
        # khớp", nó là "không có dữ liệu", và `covered` mới là chỗ nói điều đó.
        rows = [i for i, k in enumerate(ids) if (text_of.get(k) or "").strip()]
        self._rows = rows
        # BỎ DẤU cả hai phía. `bm25.py` cố tình KHÔNG tự bỏ dấu (một cài đặt duy nhất,
        # ở `src/text/fold.py`), nên chỗ gọi phải làm — và nếu quên thì token bị xé vụn
        # và điểm gần như vô nghĩa. Chính `bm25.looks_unfolded` đã bắt được lỗi này khi
        # tôi đưa truy vấn còn dấu vào: ba nguồn BM25 chấm ra trung vị hạng 14k–47k.
        kk, bb = BM25_PARAMS.get(name, (DEFAULT_K1, DEFAULT_B))
        if k1 is not None:
            kk = k1
        if b is not None:
            bb = b
        self._bm25 = (Bm25Index.build(((str(i), strip_diacritics(text_of[ids[i]]))
                                       for i in rows), k1=kk, b=bb)
                      if rows else None)

    def score(self, query_text: str) -> SourceScores:
        s = np.zeros(self._n, dtype=np.float32)
        cov = np.zeros(self._n, dtype=bool)
        if self._rows:
            cov[self._rows] = True
        q = strip_diacritics(query_text)
        if self._bm25 is not None and q.strip():
            for hit in self._bm25.search(q):
                s[int(hit.doc_id)] = hit.score
        return SourceScores(self.name, s, cov)


class AsrSource:
    """
    Lời nói, gán cho khung theo CỬA SỔ THỜI GIAN — xem docstring module.

    `frame_ms` là mốc thời gian của từng khung, lấy từ metadata của bộ cắt
    (`frame_idx / fps × 1000`). Không tự suy từ `pts_time`: Tầng 0 đã chốt δ = 0 từ
    `map-keyframes`, và tự suy lại là dựng nguồn sự thật thứ hai.
    """

    name = "asr"

    def __init__(self, ids: Sequence[FrameKey], frame_ms: Mapping[FrameKey, float],
                 segments: Mapping[str, Sequence[tuple[int, int, str]]],
                 window_ms: int = ASR_WINDOW_MS) -> None:
        text_of: dict[FrameKey, str] = {}
        by_video: dict[str, list[tuple[int, int, str]]] = {
            v: sorted(sg) for v, sg in segments.items()
        }
        for k in ids:
            ms = frame_ms.get(k)
            segs = by_video.get(k[0])
            if ms is None or not segs:
                continue
            lo, hi = ms - window_ms, ms + window_ms
            parts = [t for a, b, t in segs if b >= lo and a <= hi]
            if parts:
                text_of[k] = " ".join(parts)
        self._inner = TextSource(self.name, ids, text_of)

    def score(self, query_text: str) -> SourceScores:
        return self._inner.score(query_text)


# =============================================================================
# ĐỌC DỮ LIỆU THẬT — tách khỏi phần lõi để lõi test được không cần file
# =============================================================================


def load_ocr_text(path: str | Path, field: str = "text_ascii_folded") -> dict[FrameKey, str]:
    """
    `{(video, n) → chuỗi}` từ `data/OCR/ocr.jsonl`.

    Mặc định lấy `text_ascii_folded`: `bm25.tokenize` làm việc trên văn bản đã bỏ dấu,
    và file OCR đã có sẵn trường đó — dùng nó thay vì bỏ dấu lại là tránh có **hai**
    cài đặt bỏ dấu, đúng lý do `bm25.py` từ chối tự bỏ dấu.
    """
    out: dict[FrameKey, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            t = (d.get(field) or "").strip()
            if t:
                out[(d["video_id"], int(d["n"]))] = t
    return out


def load_shot_bounds(meta_glob: str = "data/Framme/*/metadata/*.csv"
                     ) -> dict[FrameKey, tuple[int, int]]:
    """
    `{(video, n) → (shot_start_frame, shot_end_frame)}` — biên CẢNH thật của keyframe.

    Metadata TransNetV2 mang sẵn hai cột này. Dùng chúng thay cho ước lượng "nửa khe
    tới keyframe hàng xóm" ở ⑦ vì hai lý do:

    * **Đúng hơn.** Mốc ngữ nghĩa mà một keyframe đại diện nằm TRONG cảnh của nó. Nửa
      khe có thể vắt qua ranh giới cảnh, tức rải sang nội dung khác hẳn.
    * **Hợp lệ.** Với keyframe cuối video, "nửa khe" phải đoán bằng một hằng số và có
      thể vượt quá số khung thật — bài nộp khi đó SAI ĐỊNH DẠNG. `writer.validate_all`
      bắt được: *"frame_id 16994 ≥ số khung hình 16993"*.
    """
    out: dict[FrameKey, tuple[int, int]] = {}
    for mp in sorted(Path().glob(meta_glob)):
        lines = open(mp, encoding="utf-8-sig").readlines()
        i = next(j for j, l in enumerate(lines) if l.strip())
        for r in csv.DictReader(lines[i:]):
            if r.get("shot_start_frame") and r.get("shot_end_frame"):
                out[(mp.stem, int(r["n"]))] = (int(r["shot_start_frame"]),
                                               int(r["shot_end_frame"]))
    return out


def load_video_last_frame(meta_glob: str = "data/Framme/*/metadata/*.csv"
                          ) -> dict[str, int]:
    """`{video → khung cuối cùng biết được}` = max `shot_end_frame`. Chặn trên khi nộp."""
    out: dict[str, int] = {}
    for mp in sorted(Path().glob(meta_glob)):
        lines = open(mp, encoding="utf-8-sig").readlines()
        i = next(j for j, l in enumerate(lines) if l.strip())
        m = 0
        for r in csv.DictReader(lines[i:]):
            if r.get("shot_end_frame"):
                m = max(m, int(r["shot_end_frame"]))
        if m:
            out[mp.stem] = m
    return out


def load_shot_id(path: str | Path = "data/OCR/ocr.jsonl") -> dict[FrameKey, int]:
    """
    `{(video, n) → shot_id}` — ranh giới cảnh TransNetV2, lấy từ file OCR.

    Vì sao lấy ở đây chứ không ở metadata: file OCR đã mang sẵn `shot_id` cho **đủ
    173.426 khung**, còn metadata thì không có trường này. Một nguồn sự thật, không
    phải tiện tay.

    Dùng cho khử trùng ở ⑦: nhiều khung cùng một cảnh nằm cùng một cửa sổ đáp án, nên
    nộp cả cụm là mua đúng MỘT cơ hội bằng nhiều slot. [ĐO] khử trùng theo cảnh
    **+2,0pp Final, thắng 23/thua 0, p < 0,0001** trên 226 truy vấn.
    """
    out: dict[FrameKey, int] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("shot_id")
            if sid is not None:
                out[(d["video_id"], int(d["n"]))] = int(sid)
    return out


def load_object_text(root: str | Path,
                     fields: Sequence[str] = ("layout_folded", "count_folded",
                                              "classes_folded")) -> dict[FrameKey, str]:
    """
    `{(video, n) → chuỗi}` từ `data/objects-full`, nối ba trường đã bỏ dấu sẵn.

    Ba trường có **phạm vi khác nhau** và bổ sung cho nhau: `layout_folded` nói vị trí
    (giới hạn 24 vật), `count_folded` nói số lượng (bỏ lớp chỉ 1 cá thể), còn
    `classes_folded` liệt kê có mặt **không giới hạn**. Nối cả ba thì không lớp nào
    vô hình — giao của hai chỗ bị cắt kia chính là lỗ mà `classes_folded` sinh ra để bịt.
    """
    out: dict[FrameKey, str] = {}
    for vd in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for f in vd.glob("*.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            t = " ".join(x for x in (d.get(k) or "" for k in fields) if x).strip()
            if t:
                out[(vd.name, int(f.stem))] = t
    return out


def load_asr_segments(root: str | Path) -> dict[str, list[tuple[int, int, str]]]:
    """`{video → [(start_ms, end_ms, text)]}` từ `data/ASR/*/results/*.json`."""
    out: dict[str, list[tuple[int, int, str]]] = {}
    for f in sorted(Path(root).glob("*/results/*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        segs = []
        for s in d.get("segments", []):
            t = (s.get("text") or "").strip()
            if t:
                segs.append((int(s["start_ms"]), int(s["end_ms"]), t))
        if segs:
            out[d.get("video_id", f.stem)] = segs
    return out


def load_frame_idx(meta_glob: str = "data/Framme/*/metadata/*.csv") -> dict[FrameKey, int]:
    """
    `{(video, n) → frame_idx}` — **cầu bắt buộc giữa truy xuất và bài nộp**.

    =========================================================================
    `n` KHÔNG PHẢI `frame_idx`, VÀ NHẦM LÀ HỎNG TOÀN BỘ BÀI NỘP
    =========================================================================

    `n` là **số thứ tự keyframe** trong video (1, 2, 3…). `frame_idx` là **số khung
    thật** trong video gốc. Luật thi chấm theo `frame_idx`.

    Đo trên một khung thật: `L21_V001` keyframe `n = 280` có `frame_idx = 18358` —
    **lệch 18.078**. Nộp `n` thay `frame_idx` làm MỌI câu trả lời sai.

    Và `writer.py` **không bắt được**: nó kiểm `0 ≤ frame_id < số khung của video`, mà
    `n` luôn thoả điều kiện đó (keyframe luôn ít hơn khung). Validator xanh, bài nộp
    sai, không có tín hiệu nào. Đúng lớp lỗi mà Tầng 0 tồn tại để chặn, chỉ khác cửa.

    Nên mọi đường đi từ chỉ mục (khoá theo `n`) ra bài nộp PHẢI đi qua hàm này.
    """
    import csv

    out: dict[FrameKey, int] = {}
    for p in sorted(Path().glob(meta_glob)):
        with open(p, newline="", encoding="utf-8-sig") as fh:
            lines = fh.readlines()
        i = 0
        while i < len(lines) and not lines[i].strip():
            i += 1
        for r in csv.DictReader(lines[i:]):
            if r.get("frame_idx"):
                out[(p.stem, int(r["n"]))] = int(r["frame_idx"])
    return out


def load_frame_ms(meta_glob: str = "data/Framme/*/metadata/*.csv") -> dict[FrameKey, float]:
    """
    `{(video, n) → mốc thời gian ms}` — `frame_idx / fps × 1000`, KHÔNG phải `n / fps`.

    Dùng `frame_idx` vì đó là số khung thật; `n` chỉ là số thứ tự keyframe và khoảng
    cách giữa hai `n` liền nhau dao động **p10 = 19 tới p90 = 105 khung**. Lấy `n` làm
    trục thời gian là coi mọi bước thưa như nhau.
    """
    import csv

    out: dict[FrameKey, float] = {}
    for p in sorted(Path().glob(meta_glob)):
        with open(p, newline="", encoding="utf-8-sig") as fh:
            lines = fh.readlines()
        i = 0
        while i < len(lines) and not lines[i].strip():
            i += 1
        for r in csv.DictReader(lines[i:]):
            if r.get("frame_idx") and r.get("fps"):
                out[(p.stem, int(r["n"]))] = int(r["frame_idx"]) / float(r["fps"]) * 1000.0
    return out
