"""
Chỉ mục vector PHẲNG — một ma trận, không database
===================================================

    data/embed/
      emb.npy      float16 (N, 1024)  173.426 × 1024 = 355 MB, đã L2-chuẩn hoá
                   lưu ĐỦ chiều; cắt Matryoshka xảy ra lúc ĐỌC, xem `load_flat_index`
      ids.npy        (N, 2)   [video_id, n] — khoá join, `n` là SỐ THỨ TỰ keyframe
      frame_idx.npy  (N,)     số khung THẬT trong video — thứ luật thi chấm
      ranges.json    {video_id: [lo, hi]}

Tìm kiếm là **một phép nhân ma trận**: `emb @ q`. Không Milvus, không FAISS, không ANN.

=============================================================================
VÌ SAO KHÔNG DÙNG ANN — ĐO, KHÔNG PHẢI Ý THÍCH
=============================================================================

`IndexFlatIP` của FAISS **giống hệt** `emb @ q`: cùng phép toán, cùng kết quả, chỉ khác
viết bằng C++. FAISS chỉ thật sự khác khi bật chế độ xấp xỉ (`IVF`/`HNSW`/`PQ`), và khi
đó nó đổi **recall** lấy tốc độ.

[ĐO] 173.426 × 512 fp32: **5,4 ms/truy vấn** (32,6 GFLOP/s), cả 300 truy vấn thi hết
**1,63 s**. Không có độ trễ nào để mua, nên đổi recall lấy 5 mili-giây là đổi lỗ — nhất
là khi khoản recall mất đi **không đo được** nếu chưa có ground truth.

Ngoại suy tuyến tính: ~10 triệu vector mới thành 311 ms/truy vấn và ~20 GB fp32. Đó là
lúc ANN đúng. Ta đang ở 173 nghìn, kém mốc đó 57 lần.

=============================================================================
THỨ TỰ HÀNG LÀ MỘT PHẦN CỦA THIẾT KẾ, KHÔNG PHẢI CHI TIẾT CÀI ĐẶT
=============================================================================

Hàng sắp theo `(video_id, n)` ⟹ mỗi video là **một lát liên tục** `emb[lo:hi]`.

DANTE chạy DP **theo từng video** và cần đúng lát đó. Nếu hàng xếp lộn xộn thì mỗi video
phải `gather` theo danh sách chỉ số — chậm hơn, tốn bộ nhớ hơn, và mở đường cho lỗi lệch
thứ tự. `ranges.json` chỉ là chỉ số của các lát ấy, nên nó KHÔNG phải nguồn sự thật thứ
hai: `build_flat_index` tính nó ra từ chính `ids`.

=============================================================================
`n` KHÔNG PHẢI `frame_idx` — VÀ CHỈ MỤC PHẢI MANG CẢ HAI
=============================================================================

`n` là **số thứ tự keyframe** (1, 2, 3…). `frame_idx` là **số khung thật** trong video.
Luật thi chấm theo `frame_idx`.

[ĐO] **0/173.426** khung có `n == frame_idx`. Lệch trung vị **5.267**, lớn nhất
**68.464**. `L21_V001` keyframe `n = 280` có `frame_idx = 18358`.

Nộp nhầm `n` làm **mọi câu sai**, và `writer.py` không bắt được vì nó chỉ kiểm
`0 ≤ frame_id < số khung video` — mà `n` luôn thoả (keyframe luôn ít hơn khung).

Nên chỉ mục **tự mang `frame_idx`**, thay vì để mỗi chỗ dùng tự nhớ tra bảng. Quên tra
là lỗi im lặng; không thể quên nếu dữ liệu đã đi kèm. `answer()` trả thẳng dạng nộp bài.

=============================================================================
HAI MẢNG SONG SONG LỆCH NHAU LÀ LỖI KHÔNG BAO GIỜ TỰ BÁO
=============================================================================

`emb[i]` phải ứng với `ids[i]`. Lệch một hàng thì mọi truy vấn trả về khung SAI mà không
có exception nào — điểm vẫn là số thực hợp lệ, thứ hạng vẫn sắp được.

Đây không phải rủi ro lý thuyết. Đúng lớp lỗi này vừa xảy ra ở chỉ mục detection:
`detection_boxes` dài 100 còn `regions` dài 16, ai `zip()` hai mảng đó ghép hộp thứ 3 với
vùng của một vật khác — im lặng hoàn toàn. Nên `load_flat_index` **kiểm và ném lỗi**,
chứ không tin.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.ingestion.jina_encoder import is_normalized, truncate_and_normalize

logger = logging.getLogger(__name__)

EMB_FILE = "emb.npy"
IDS_FILE = "ids.npy"
FIDX_FILE = "frame_idx.npy"
RANGES_FILE = "ranges.json"


@dataclass(frozen=True)
class FlatIndex:
    """
    Chỉ mục phẳng đã nạp vào RAM.

    `emb` là **fp32** dù trên đĩa là fp16: BLAS chạy fp32 nhanh hơn nhiều, và 355 MB
    vẫn là con số không đáng bàn. fp16 chỉ để tiết kiệm đĩa và băng thông đọc.
    """

    emb: np.ndarray                      # (N, D) fp32, L2-chuẩn hoá
    ids: list[tuple[str, int]]           # [(video_id, n)] — `n` là số THỨ TỰ keyframe
    ranges: dict[str, tuple[int, int]]   # {video_id: (lo, hi)} — nửa mở [lo, hi)
    frame_idx: np.ndarray                # (N,) int32 — số khung THẬT, dùng để NỘP BÀI

    def answer(self, row: int) -> tuple[str, int]:
        """
        `(video_id, frame_idx)` — **dạng duy nhất được phép đưa ra bài nộp**.

        Dùng hàm này thay vì `ids[row]`: `ids` mang `n` (số thứ tự keyframe), còn luật
        thi chấm `frame_idx`. Đo được 0/173.426 khung có hai giá trị đó bằng nhau, nên
        nhầm là sai mọi câu — và validator không bắt được.
        """
        return self.ids[row][0], int(self.frame_idx[row])

    def answer_path(self, rows: Sequence[int]) -> tuple[str, tuple[int, ...]]:
        """
        `(video_id, (frame_idx…))` cho MỘT câu trả lời — dạng `TaskSubmission.answers`.

        `N = 1` cho KIS/QA, `N` cho TRAKE. Đây là đường ra bài nộp **duy nhất nên
        dùng**: bên gọi truyền chỉ số hàng, không bao giờ gõ số khung, nên không có cửa
        nào để nhầm `n` với `frame_idx`.

        Lưới an toàn `valid_frames` của `writer.py` chỉ bắt được **97,4%** ca nhầm — đo
        được **4.584/173.426 = 2,6%** khung có `n` trùng đúng một `frame_idx` hợp lệ của
        cùng video đó (n từ 3 tới 720, trung vị 83). Nên lưới là lớp thứ hai, còn lớp
        thứ nhất là **không tạo ra cơ hội nhầm**.

        Raises:
            ValueError: các hàng thuộc video khác nhau, hoặc `frame_idx` không tăng dần
                (TRAKE bắt buộc tăng dần — `writer` cũng kiểm, nhưng chặn sớm thì thông
                báo lỗi chỉ đúng vào chỗ sai).
        """
        if not rows:
            raise ValueError("cần ít nhất một hàng")
        vids = {self.ids[r][0] for r in rows}
        if len(vids) != 1:
            raise ValueError(f"một câu trả lời phải cùng MỘT video, nhận {sorted(vids)}")
        fr = tuple(int(self.frame_idx[r]) for r in rows)
        if list(fr) != sorted(fr):
            raise ValueError(f"các mốc phải tăng dần theo thời gian, nhận {fr}")
        return vids.pop(), fr

    @property
    def n_frames(self) -> int:
        return self.emb.shape[0]

    @property
    def dim(self) -> int:
        return self.emb.shape[1]

    def search(self, query: np.ndarray, top_k: int = 100) -> list[tuple[int, float]]:
        """
        `[(chỉ_số_hàng, điểm)]` sắp giảm dần — quét phẳng toàn bộ, kết quả CHÍNH XÁC.

        `query` phải cùng số chiều và đã chuẩn hoá; khi đó tích vô hướng chính là cosine.

        Dùng `argpartition` chứ không `argsort` toàn mảng: chỉ cần top-k thì sắp đủ
        173.426 phần tử là làm thừa `O(N log N)` trong khi `O(N + k log k)` là đủ.
        """
        q = np.asarray(query, dtype=np.float32).ravel()
        if q.shape[0] != self.dim:
            raise ValueError(f"truy vấn {q.shape[0]} chiều ≠ chỉ mục {self.dim} chiều")
        if top_k <= 0:
            raise ValueError(f"top_k phải dương, nhận {top_k}")
        scores = self.emb @ q
        k = min(top_k, scores.shape[0])
        if k == 0:
            return []
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part])]
        return [(int(i), float(scores[i])) for i in order]

    def video_slice(self, video_id: str) -> np.ndarray:
        """
        `emb[lo:hi]` của một video — **view**, không sao chép. Đầu vào của DANTE.

        Raises:
            KeyError: video không có trong chỉ mục. Ném lỗi chứ không trả mảng rỗng:
                mảng rỗng làm DP chạy tiếp và cho điểm `None` như thể video đó không
                khớp, giấu mất chuyện chỉ mục dựng thiếu.
        """
        if video_id not in self.ranges:
            raise KeyError(f"{video_id!r} không có trong chỉ mục ({len(self.ranges)} video)")
        lo, hi = self.ranges[video_id]
        return self.emb[lo:hi]

    def frames_of(self, video_id: str) -> list[int]:
        """Danh sách `n` của một video, tăng dần — trục thời gian cho DP."""
        lo, hi = self.ranges[video_id]
        return [n for _, n in self.ids[lo:hi]]


def build_flat_index(rows: Iterable[tuple[str, int, np.ndarray]],
                     frame_idx_of: Mapping[tuple[str, int], int] | None = None) -> FlatIndex:
    """
    `[(video_id, n, vector)]` + `{(video, n) → frame_idx}` → `FlatIndex`, đã sắp và kiểm.

    Tự sắp theo `(video_id, n)` thay vì đòi bên gọi sắp sẵn: thứ tự là bất biến của
    ĐỊNH DẠNG, nên nơi duy nhất bảo đảm nó phải là nơi dựng ra định dạng.

    Raises:
        ValueError: trùng `(video_id, n)`, lệch số chiều, hoặc vector chưa chuẩn hoá.
    """
    items = list(rows)
    if not items:
        return FlatIndex(np.zeros((0, 0), dtype=np.float32), [], {},
                         np.zeros(0, dtype=np.int32))

    dims = {np.asarray(v).ravel().shape[0] for _, _, v in items}
    if len(dims) != 1:
        raise ValueError(f"các vector lệch số chiều: {sorted(dims)}")

    items.sort(key=lambda r: (r[0], r[1]))
    ids = [(v, int(n)) for v, n, _ in items]
    if len(set(ids)) != len(ids):
        dup = next(k for k in ids if ids.count(k) > 1)
        raise ValueError(f"trùng khoá {dup} — hai vector cho cùng một khung")

    emb = np.stack([np.asarray(v, dtype=np.float32).ravel() for _, _, v in items])
    if not is_normalized(emb):
        raise ValueError(
            "vector chưa L2-chuẩn hoá — tích vô hướng sẽ KHÔNG phải cosine, và độ dài "
            "vector (vô nghĩa về ngữ nghĩa) sẽ lẻn vào điểm số"
        )

    if frame_idx_of is None:
        raise ValueError(
            "thiếu `frame_idx_of` — chỉ mục PHẢI mang số khung thật, nếu không mọi "
            "đường ra bài nộp đều phải tự nhớ tra bảng, và quên là lỗi im lặng"
        )
    missing = [k for k in ids if k not in frame_idx_of]
    if missing:
        raise ValueError(f"thiếu frame_idx cho {len(missing)} khung, vd {missing[:3]}")
    fidx = np.array([frame_idx_of[k] for k in ids], dtype=np.int32)

    ranges: dict[str, tuple[int, int]] = {}
    for i, (vid, _) in enumerate(ids):
        lo, hi = ranges.get(vid, (i, i))
        ranges[vid] = (lo, i + 1)
    return FlatIndex(emb, ids, ranges, fidx)


def save_flat_index(out_dir: str | Path, index: FlatIndex) -> None:
    """Ghi ra đĩa. `emb` xuống fp16 (một nửa dung lượng), `ids` giữ nguyên văn."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / EMB_FILE, index.emb.astype(np.float16))
    np.save(d / IDS_FILE, np.array([[v, str(n)] for v, n in index.ids], dtype=object),
            allow_pickle=True)
    np.save(d / FIDX_FILE, index.frame_idx.astype(np.int32))
    (d / RANGES_FILE).write_text(
        json.dumps({v: list(r) for v, r in index.ranges.items()}, ensure_ascii=False),
        encoding="utf-8")


def load_flat_index(out_dir: str | Path, dim: int | None = None) -> FlatIndex:
    """
    Đọc lại và **KIỂM MỌI BẤT BIẾN**. Xem docstring module về vì sao phải kiểm.

    `dim` cắt Matryoshka **lúc đọc**: chỉ mục lưu đủ 1024 chiều, còn phép tìm kiếm
    chạy ở số chiều ta chọn. Đây mới là chỗ đúng để cắt — cắt lúc mã hoá là vứt luôn
    quyền chọn, và lấy lại thì phải mã hoá lại toàn bộ 173.426 khung.

    Raises:
        ValueError: số hàng ≠ số id · lát video không liên tục · `ranges` không khớp
            `ids` · vector không còn chuẩn hoá.
    """
    d = Path(out_dir)
    emb = np.load(d / EMB_FILE).astype(np.float32)
    raw = np.load(d / IDS_FILE, allow_pickle=True)
    ids = [(str(v), int(n)) for v, n in raw]
    ranges = {v: (int(a), int(b))
              for v, (a, b) in json.loads((d / RANGES_FILE).read_text(encoding="utf-8")).items()}

    if emb.shape[0] != len(ids):
        raise ValueError(
            f"{emb.shape[0]} hàng embedding ≠ {len(ids)} id — hai mảng song song đã "
            f"lệch, mọi truy vấn sẽ trả khung SAI mà không báo lỗi"
        )
    if ids != sorted(ids):
        raise ValueError("ids chưa sắp theo (video_id, n) — lát video sẽ không liên tục")
    if not is_normalized(emb):
        raise ValueError("vector đọc lên không còn chuẩn hoá")

    rebuilt: dict[str, tuple[int, int]] = {}
    for i, (vid, _) in enumerate(ids):
        lo, _hi = rebuilt.get(vid, (i, i))
        rebuilt[vid] = (lo, i + 1)
    if rebuilt != ranges:
        bad = [v for v in set(rebuilt) | set(ranges) if rebuilt.get(v) != ranges.get(v)]
        raise ValueError(f"ranges.json không khớp ids ở {len(bad)} video, vd {bad[:3]}")

    fp = d / FIDX_FILE
    if not fp.exists():
        raise ValueError(
            f"thiếu {FIDX_FILE} — chỉ mục cũ chỉ có `n`. Dựng lại bằng 02_build_index.py: "
            f"nộp `n` thay `frame_idx` làm sai MỌI câu và validator không bắt được"
        )
    fidx = np.load(fp).astype(np.int32)
    if fidx.shape[0] != len(ids):
        raise ValueError(f"{fidx.shape[0]} frame_idx ≠ {len(ids)} id — hai mảng đã lệch")

    if dim is not None:
        emb = truncate_and_normalize(emb, dim)   # cắt RỒI chuẩn hoá — thứ tự bắt buộc
    return FlatIndex(emb, ids, ranges, fidx)


def save_video_shard(
    shard_dir: str | Path, video_id: str, ns: Sequence[int], mat: np.ndarray
) -> tuple[Path, Path]:
    """
    Ghi một video ra `{video_id}.npy` + `.json`. Trả `(đường .npy, đường .json)`.

    =========================================================================
    HAI CHI TIẾT NHỎ, MỖI CÁI TỪNG LÀ MỘT BUG
    =========================================================================

    **1. Tên tạm phải kết thúc bằng `.npy`.** `np.save` **tự thêm** `.npy` khi tên chưa
    có đuôi đó — nên `L21_V001.npy.tmp` biến thành `L21_V001.npy.tmp.npy`, và
    `replace()` nổ `FileNotFoundError`. Bug này từng sống trong script Modal và **sống
    sót qua cả benchmark lẫn test đơn vị**: nhánh benchmark `return` trước khi tới đoạn
    ghi, còn test thì tự tạo file bằng tên đúng chuẩn nên không đụng tên tạm. Đó là lý
    do hàm này tồn tại — logic ghi phải nằm ở chỗ test được, không nằm trong script.

    **2. `.npy` ghi TRƯỚC `.json`.** `video_is_encoded` đòi **cả hai**, nên chết giữa hai
    lệnh thì thiếu `.json` ⟹ video được làm lại. Thứ tự ngược lại để `.json` đủ bên cạnh
    `.npy` cắt dở, và phải tới điều kiện đếm hàng mới bắt được — muộn hơn một nhịp.

    Raises:
        ValueError: số hàng ≠ số khung — chặn tại chỗ ghi thay vì để lệch xuống hạ nguồn.
    """
    if mat.shape[0] != len(ns):
        raise ValueError(f"{mat.shape[0]} vector ≠ {len(ns)} khung cho {video_id}")
    d = Path(shard_dir)
    d.mkdir(parents=True, exist_ok=True)
    vp, jp = d / f"{video_id}.npy", d / f"{video_id}.json"
    tmp = vp.with_name(vp.name + ".tmp.npy")
    np.save(tmp, np.asarray(mat, dtype=np.float16))
    tmp.replace(vp)
    jp.write_text(json.dumps(list(ns)), encoding="utf-8")
    return vp, jp


def video_is_encoded(
    shard_dir: str | Path, video_id: str, expected_ns: Sequence[int], dim: int
) -> tuple[bool, str]:
    """
    `(video này đã mã hoá xong chưa, lý do)` — checkpoint cho lượt chạy trên Modal.

    =========================================================================
    "XONG" LÀ ĐỌC ĐƯỢC VÀ ĐÚNG, KHÔNG PHẢI FILE CÓ TỒN TẠI
    =========================================================================

    Container Modal là **spot**, bị thu hồi bất cứ lúc nào — kể cả giữa lúc ghi. Nên
    một checkpoint chỉ đếm file sẽ coi file cắt dở là "xong" và bỏ qua vĩnh viễn, để
    lại một lỗ trong chỉ mục mà **không có gì báo**.

    Bốn điều kiện, mỗi cái ứng với một kiểu hỏng đã lường:

        1. có cả `.npy` và `.json`      — ghi dở một trong hai
        2. `.json` khớp ĐÚNG danh sách khung, không chỉ khớp số lượng
                                        — mã hoá nhầm tập khung
        3. số hàng `.npy` = số khung    — hai mảng song song lệch
        4. số chiều đúng                — đổi `dim` giữa chừng mà không xoá cũ

    Đọc header `.npy` bằng `mmap_mode="r"` chứ không nạp cả mảng: với 873 video, nạp
    thật là đọc thừa hàng trăm MB chỉ để xem `shape`.
    """
    d = Path(shard_dir)
    vp, jp = d / f"{video_id}.npy", d / f"{video_id}.json"
    if not vp.exists() or not jp.exists():
        return False, "thiếu .npy hoặc .json"
    try:
        ns = json.loads(jp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return False, f".json không đọc được ({type(e).__name__}) — gần như chắc là cắt dở"
    if list(ns) != list(expected_ns):
        return False, f".json có {len(ns)} khung nhưng cần {len(expected_ns)}, hoặc lệch danh sách"
    try:
        arr = np.load(vp, mmap_mode="r")
    except (OSError, ValueError) as e:
        return False, f".npy không đọc được ({type(e).__name__}) — cắt dở"
    if arr.shape[0] != len(ns):
        return False, f"{arr.shape[0]} vector ≠ {len(ns)} khung"
    if arr.ndim != 2 or arr.shape[1] != dim:
        return False, f"số chiều {arr.shape[1:]} ≠ {dim}"
    return True, f"đủ {len(ns)} khung"


def pending_videos(
    shard_dir: str | Path, expected: Mapping[str, Sequence[int]], dim: int
) -> tuple[list[str], dict[str, str]]:
    """`(video CHƯA xong, {video: lý do})` — dùng cho lượt mã hoá nối tiếp trên Modal."""
    todo, why = [], {}
    for vid, ns in expected.items():
        ok, reason = video_is_encoded(shard_dir, vid, ns, dim)
        if not ok:
            todo.append(vid)
            why[vid] = reason
    return todo, why


def check_alignment(index: FlatIndex, expected: Sequence[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    Khung nào có trong `expected` mà chỉ mục THIẾU — trả danh sách, rỗng là khớp hết.

    Cùng chữ ký `check_alignment` của `ocr_store`/`layout_store` để team gọi được theo
    một khuôn duy nhất trên mọi chỉ mục.
    """
    have = set(index.ids)
    return [k for k in expected if k not in have]
