"""
Phiên tìm kiếm có phản hồi — trạng thái và vòng lặp click (tài liệu §4.1, §5, §8)
=================================================================================

MỘT QUYẾT ĐỊNH VỀ HIỆU NĂNG QUYẾT ĐỊNH CẢ THIẾT KẾ. Hợp đồng độ trễ §6.1 đòi một
vòng feedback **dưới 500 ms**. Chấm BM25 trên 609.476 khung mất hàng chục giây, nên
nếu mỗi vòng chấm lại thì không có cách nào đạt.

Nhưng Rocchio **chỉ đổi run thị giác** — câu hỏi văn bản không đổi thì BM25 không
đổi. Nên phiên **giữ sẵn đóng góp RRF của OCR/ASR** tính một lần ở vòng 0, và mỗi
vòng sau chỉ tốn:

    q′ = update(q₀, D⁺)          ~1 ms   (vài phép cộng vector)
    s  = emb @ q′                ~80 ms  (609.476 × 512)
    RRF(s) + cộng phần văn bản   ~70 ms  (một argsort 609.476)

Đây KHÔNG phải rerank trên danh sách cũ: `emb @ q′` chạm **mọi** khung, nên một
khung đang nằm hạng 400.000 vẫn có đường đi lên — đúng yêu cầu §2.

TRAKE giữ query state RIÊNG TỪNG EVENT (§8). Click cho E₂ chỉ cập nhật `q₂`; E₁/E₃
không đổi. Ràng buộc này nằm trong cấu trúc dữ liệu (`_state` khoá theo `event_id`)
chứ không nằm ở quy ước gọi hàm, nên không thể quên.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np

from src.feedback.rocchio import RocchioConfig, drift_guard, l2, update
from src.retrieval.score_matrix import rrf_normalize

__all__ = ["Feedback", "EventState", "SearchSession"]


@dataclass(frozen=True)
class Feedback:
    """Một lần click. `row` là chỉ số hàng trong chỉ mục phẳng."""

    row: int
    positive: bool
    event_id: int = 0


@dataclass
class EventState:
    """Trạng thái Rocchio của MỘT event. KIS/QA chỉ có event 0."""

    q_original: np.ndarray
    q_current: np.ndarray
    positives: list[int] = field(default_factory=list)
    negatives: list[int] = field(default_factory=list)
    #: β thực tế đang dùng — `drift_guard` có thể hạ nó xuống (§12.1).
    beta_effective: float = 0.0
    last_cos: float = 1.0


class SearchSession:
    """
    Một phiên tìm kiếm: giữ `q₀`, danh sách feedback, và bản đệm RRF văn bản.

    Args:
        index: đối tượng có `.emb (n,d)`, `.ids`, `.frame_idx`, `.n_frames`.
        text_rrf: `{modality: vector RRF đã gộp expansion}` — tính MỘT lần ở vòng 0.
        times_s: mốc thời gian từng khung, giây. Dùng dựng prototype cửa sổ.
        video_of: mã video từng khung.
    """

    def __init__(self, index, text_rrf: dict[str, np.ndarray],
                 times_s: np.ndarray, video_of: np.ndarray,
                 alpha: dict[str, float], *, rrf_k: float = 60.0,
                 cfg: RocchioConfig = RocchioConfig(), task: str = "kis",
                 query_text: str = "",
                 visual_extra: np.ndarray | None = None,
                 n_visual_run: int = 1) -> None:
        self.session_id = uuid.uuid4().hex[:12]
        self.index = index
        self.text_rrf = text_rrf
        self.times_s = np.asarray(times_s, dtype=np.float64)
        self.video_of = np.asarray(video_of)
        self.alpha = dict(alpha)
        self.rrf_k = float(rrf_k)
        self.cfg = cfg
        self.task = task
        self.query_text = query_text
        #: Đóng góp RRF của các run THỊ GIÁC MỞ RỘNG — đã nhân sẵn beta, cộng lại.
        #: Chúng TĨNH: Rocchio chỉ đổi run gốc (`q_current`), còn bản mở rộng là câu
        #: khác nên không có `q₀` để cập nhật. Tách hai phần cho rõ chứ không gộp,
        #: vì gộp rồi thì không biết phần nào Rocchio được phép động vào.
        self.visual_extra = (None if visual_extra is None
                             else np.asarray(visual_extra, dtype=np.float32))
        #: Tổng số run trong modality thị giác (gốc + mở rộng) — quyết định beta.
        self.n_visual_run = max(1, int(n_visual_run))
        self.round = 0
        self.history: list[Feedback] = []
        #: Khung operator ĐÃ CHỐT, theo thứ tự bấm. Chúng được ghim lên đầu bài nộp;
        #: 99 dòng còn lại lấy từ xếp hạng hiện tại. Lý do ghim: `R@k = max` nên dòng
        #: đầu là ô đắt nhất (trọng số 1,0 so với 0,2 ở dòng 51–100), và người vừa
        #: nhìn ảnh biết chắc hơn mọi phép xếp hạng.
        self.chot: list[tuple[str, tuple[int, ...]]] = []
        self.answer: str = ""
        #: Nhật ký xuống đĩa. `None` = không lưu (dùng trong test và khi phát lại —
        #: phát lại mà vẫn ghi thì nhật ký tự nhân đôi mỗi lần khôi phục).
        self.store = None
        self._state: dict[int, EventState] = {}
        self.latency_ms: list[float] = []

    # ── trạng thái ────────────────────────────────────────────────────────
    def khoi_tao(self, q0: np.ndarray, event_id: int = 0) -> None:
        v = l2(q0)
        self._state[event_id] = EventState(q_original=v, q_current=v,
                                           beta_effective=self.cfg.beta)

    def state(self, event_id: int = 0) -> EventState:
        if event_id not in self._state:
            raise KeyError(f"event {event_id} chưa khởi tạo — gọi `khoi_tao` trước")
        return self._state[event_id]

    # ── prototype ─────────────────────────────────────────────────────────
    def _prototype_rows(self, row: int) -> np.ndarray:
        """
        Khung gộp vào prototype quanh một click: CÙNG video và trong `radius_sec`.

        Tài liệu §3.2 ưu tiên same-shot; kho này chỉ có ranh giới shot cho 216/873
        video (46,7% khung), nên dùng đường lùi mà chính §3.2 cho phép: cửa sổ thời
        gian. Chặn `max_frames` để một cảnh dày keyframe không lấn.
        """
        v, t = self.video_of[row], self.times_s[row]
        sel = np.flatnonzero((self.video_of == v)
                             & (np.abs(self.times_s - t) <= self.cfg.radius_sec))
        if sel.size == 0:
            return np.asarray([row], dtype=np.int64)
        if sel.size > self.cfg.max_frames:
            sel = sel[np.argsort(np.abs(self.times_s[sel] - t))[:self.cfg.max_frames]]
        return sel

    def _centroid_input(self, rows: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """`(vectors, groups)` — mỗi click là MỘT nhóm, để cân bằng theo §3.2."""
        idxs, grp = [], []
        for gi, r in enumerate(rows):
            sel = self._prototype_rows(r)
            idxs.extend(sel.tolist())
            grp.extend([gi] * len(sel))
        return self.index.emb[np.asarray(idxs)], np.asarray(grp)

    # ── vòng lặp ──────────────────────────────────────────────────────────
    def _fuse(self, q: np.ndarray) -> np.ndarray:
        """
        ③ với run thị giác GỐC tính từ `q`; các run còn lại lấy từ bản đệm.

        Trong modality thị giác, beta chia ĐỀU cho `n_visual_run` run: run gốc (đã
        qua Rocchio) và các bản mở rộng tĩnh. Nhờ vậy thêm bản mở rộng KHÔNG làm
        modality thị giác tự tăng quyền vote — đúng bất biến `Σ_j β_mj = 1` của ③.
        """
        s = self.index.emb @ np.asarray(q, dtype=np.float32)
        cov = np.ones(s.shape[0], dtype=bool)
        b = 1.0 / self.n_visual_run
        vis = b * rrf_normalize(s.astype(np.float32), cov, self.rrf_k)
        if self.visual_extra is not None:
            vis = vis + self.visual_extra
        out = self.alpha["visual"] * vis
        for m, r in self.text_rrf.items():
            out = out + self.alpha[m] * r
        return out.astype(np.float32)

    def rank(self, event_id: int = 0) -> np.ndarray:
        return self._fuse(self.state(event_id).q_current)

    def feedback(self, row: int, positive: bool = True, event_id: int = 0) -> dict:
        """
        Một vòng click. Trả `{round, cos, beta, drift, latency_ms, n_pos, n_neg}`.

        Cập nhật LUÔN từ `q_original` — không nối từ `q_current`. Xem tính chất (4)
        trong `rocchio.py`: nối vòng làm trọng số `q₀` suy giảm cấp số nhân.
        """
        t0 = time.perf_counter()
        st = self.state(event_id)
        ds = st.positives if positive else st.negatives
        if row in ds:
            ds.remove(row)          # click lại = bỏ chọn, không cộng dồn trùng
        else:
            ds.append(int(row))
        self.history.append(Feedback(int(row), positive, event_id))
        self.round += 1
        if self.store is not None:
            self.store.ghi("feedback", row=int(row), positive=bool(positive),
                           event_id=int(event_id))
        self._recompute(event_id)
        dt = (time.perf_counter() - t0) * 1000.0
        self.latency_ms.append(dt)
        return {"round": self.round, "cos": st.last_cos, "beta": st.beta_effective,
                "drift": st.last_cos < self.cfg.drift_threshold,
                "latency_ms": round(dt, 1),
                "n_pos": len(st.positives), "n_neg": len(st.negatives)}

    def _recompute(self, event_id: int) -> None:
        """Tính `q′` từ `q₀` + toàn bộ feedback tích luỹ, kèm drift guard §12.1."""
        st = self.state(event_id)
        if not st.positives and not st.negatives:
            st.q_current, st.beta_effective, st.last_cos = st.q_original, self.cfg.beta, 1.0
            return
        P = G = N = NG = None
        if st.positives:
            P, G = self._centroid_input(st.positives)
        if st.negatives and self.cfg.gamma > 0:
            N, NG = self._centroid_input(st.negatives)
        beta = self.cfg.beta
        for _ in range(3):          # guard tối đa 3 lần hạ β, rồi chấp nhận
            cfg = RocchioConfig(alpha=self.cfg.alpha, beta=beta,
                                gamma=self.cfg.gamma if N is not None else 0.0,
                                radius_sec=self.cfg.radius_sec,
                                max_frames=self.cfg.max_frames,
                                balance_by_shot=self.cfg.balance_by_shot,
                                drift_threshold=self.cfg.drift_threshold)
            q1 = update(st.q_original, P, N, cfg=cfg, pos_groups=G, neg_groups=NG)
            troi, c = drift_guard(st.q_original, q1, cfg)
            if not troi:
                break
            beta *= 0.5
        st.q_current, st.beta_effective, st.last_cos = q1, beta, c

    def undo(self) -> bool:
        """Bỏ lần click gần nhất. Trả `False` nếu không còn gì để bỏ."""
        if not self.history:
            return False
        f = self.history.pop()
        st = self.state(f.event_id)
        ds = st.positives if f.positive else st.negatives
        if f.row in ds:
            ds.remove(f.row)
        self.round = max(0, self.round - 1)
        self._recompute(f.event_id)
        if self.store is not None:
            self.store.ghi("undo")
        return True

    def reset(self) -> None:
        """Về `q_original` cho MỌI event. Tài liệu §5.3 gọi đây là thao tác bắt buộc."""
        self.history.clear()
        self.round = 0
        if self.store is not None:
            self.store.ghi("reset")
        for st in self._state.values():
            st.positives.clear()
            st.negatives.clear()
            st.q_current = st.q_original
            st.beta_effective = self.cfg.beta
            st.last_cos = 1.0

    # ── chốt đáp án ───────────────────────────────────────────────────────
    def submit(self, row: int) -> int:
        """Ghim một khung lên đầu bài nộp. Bấm lại = bỏ ghim. Trả số khung đã chốt."""
        # 🔴 SO SÁNH PHẢI CÙNG KIỂU VỚI THỨ ĐƯỢC LƯU. Bản trước so `(video, int)`
        # nhưng lưu `(video, (int,))`, nên `in` KHÔNG BAO GIỜ khớp: bấm lại chỉ cộng
        # thêm bản sao, không bao giờ bỏ ghim. [ĐO] một phiên thử tích 14 mục với
        # `10877` lặp 3 lần — và mỗi bản sao là một dòng bài nộp vứt đi, vì thể lệ
        # chấm `R@k = max` nên hai dòng giống hệt mua đúng MỘT cơ hội.
        key = (str(self.video_of[row]), (int(self.index.frame_idx[row]),))
        if key in self.chot:
            self.chot.remove(key)
        else:
            self.chot.append(key)
        if self.store is not None:
            self.store.ghi("submit", row=int(row))
        return len(self.chot)

    def dat_answer(self, answer: str) -> str:
        """Đặt đáp án Q&A, có ghi nhật ký. Gán thẳng `.answer` thì mất khi chết."""
        self.answer = str(answer)[:100]
        if self.store is not None:
            self.store.ghi("answer", answer=self.answer)
        return self.answer

    def dong_nop(self, top_k: int = 100, event_id: int = 0
                 ) -> list[tuple[str, tuple[int, ...]]]:
        """
        `top_k` dòng: khung đã chốt trước, rồi xếp hạng hiện tại lấp phần còn lại.

        Khử trùng `(video, frame_idx)` — thể lệ chấm `R@k = max`, nên hai dòng giống
        hệt mua đúng MỘT cơ hội và dòng thứ hai là ô vứt đi.
        """
        out: list[tuple[str, tuple[int, ...]]] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()
        for x in self.chot:
            if x not in seen:
                seen.add(x)
                out.append(x)
        sc = self.rank(event_id)
        for r in np.argsort(-sc, kind="stable").tolist():
            if len(out) >= top_k:
                break
            x = (str(self.video_of[r]), (int(self.index.frame_idx[r]),))
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out[:top_k]
