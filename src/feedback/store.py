"""
Lưu phiên xuống đĩa — chống mất việc khi server chết (tài liệu §12, §14)
========================================================================

Ngày thi có 2h30 và không có lần hai. Một tiến trình chết sau khi operator đã click
20 câu là mất 20 câu, và không có cách nào dựng lại từ trí nhớ.

    data/rf_sessions/<session_id>/
        meta.json      truy vấn · loại đề · alpha · cfg          ← ghi MỘT lần
        vectors.npz    q_original mỗi event · text_rrf mỗi nguồn ← ghi MỘT lần
        events.jsonl   một dòng mỗi thao tác                     ← NỐI THÊM

VÌ SAO TÁCH LÀM HAI. Hợp đồng §6.1 đòi vòng feedback dưới 500 ms. `text_rrf` là
609.476 × 2 số thực = 4,9 MB; ghi lại nó mỗi vòng là tự phá hợp đồng. Nhưng nó
KHÔNG đổi trong suốt phiên — câu hỏi văn bản có đổi đâu. Nên nó thuộc phần ghi một
lần, còn mỗi vòng chỉ nối một dòng JSON vài trăm byte.

VÌ SAO LÀ NHẬT KÝ NỐI THÊM, KHÔNG PHẢI ẢNH CHỤP TRẠNG THÁI. Ghi đè một file trạng
thái có cửa sổ chết người: tiến trình chết giữa lúc ghi thì file cũ đã mất mà file
mới chưa xong — mất SẠCH thay vì mất một thao tác. Nối thêm thì bản ghi cũ không
bao giờ bị đụng. Đổi lại, dòng cuối có thể cụt; `phat_lai` bỏ qua dòng hỏng thay
vì nổ, vì mất một click tốt hơn mất cả phiên.

Và nó cho luôn thứ §14 đòi — *"log đủ để replay một session"* — mà không phải viết
thêm hệ thống log thứ hai.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

__all__ = ["SessionStore", "liet_ke"]

GOC = Path("data/rf_sessions")


class SessionStore:
    """Nhật ký một phiên. `ghi_*` gọi từ `SearchSession`; `phat_lai` dựng lại."""

    def __init__(self, session_id: str, goc: str | Path = GOC) -> None:
        self.dir = Path(goc) / session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log = self.dir / "events.jsonl"

    # ── phần ghi MỘT lần ──────────────────────────────────────────────────
    def khoi_tao(self, meta: dict, q_original: dict[int, np.ndarray],
                 text_rrf: dict[str, np.ndarray]) -> None:
        (self.dir / "meta.json").write_text(
            json.dumps({**meta, "created_at": time.time()}, ensure_ascii=False),
            encoding="utf-8")
        np.savez(self.dir / "vectors.npz",
                 **{f"q{e}": v for e, v in q_original.items()},
                 **{f"t_{m}": v for m, v in text_rrf.items()})

    # ── phần NỐI THÊM ─────────────────────────────────────────────────────
    def ghi(self, op: str, **kw) -> None:
        """
        Nối một thao tác. `flush` + `fsync` để nó thật sự nằm trên đĩa TRƯỚC khi
        hàm trả về — không có bước đó thì dữ liệu còn trong bộ đệm của OS và một
        cú kill vẫn mất nó, tức lưu mà như không lưu.
        """
        with open(self._log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": time.time(), "op": op, **kw},
                                ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # ── đọc lại ───────────────────────────────────────────────────────────
    def doc(self) -> tuple[dict, dict[int, np.ndarray], dict[str, np.ndarray], list[dict]]:
        meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        z = np.load(self.dir / "vectors.npz")
        q0 = {int(k[1:]): z[k] for k in z.files if k.startswith("q")}
        txt = {k[2:]: z[k] for k in z.files if k.startswith("t_")}
        return meta, q0, txt, self._doc_events()

    def _doc_events(self) -> list[dict]:
        """Bỏ qua dòng hỏng — đó là dòng đang ghi dở lúc tiến trình chết."""
        if not self._log.is_file():
            return []
        out, hong = [], 0
        for line in self._log.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                hong += 1
        if hong:
            print(f"⚠ {self.dir.name}: bỏ {hong} dòng nhật ký hỏng (ghi dở khi chết)")
        return out


def phat_lai(ses, events: list[dict]) -> None:
    """
    Áp lại nhật ký lên một `SearchSession` đã dựng từ `meta` + `vectors`.

    Phát lại theo THAO TÁC chứ không gán thẳng trạng thái cuối: `feedback` có ngữ
    nghĩa bật/tắt (bấm lại là bỏ chọn) và `undo` phụ thuộc lịch sử, nên chỉ có chạy
    lại đúng thứ tự mới ra đúng trạng thái. Gán thẳng sẽ đúng ở ca dễ và sai ở ca
    có undo — đúng ca người ta cần lúc hoảng.
    """
    for e in events:
        op = e.get("op")
        if op == "feedback":
            ses.feedback(int(e["row"]), bool(e.get("positive", True)),
                         int(e.get("event_id", 0)))
        elif op == "undo":
            ses.undo()
        elif op == "reset":
            ses.reset()
        elif op == "submit":
            ses.submit(int(e["row"]))
        elif op == "answer":
            ses.answer = str(e.get("answer", ""))[:100]


def liet_ke(goc: str | Path = GOC) -> list[dict]:
    """Mọi phiên trên đĩa, mới nhất trước — để UI cho chọn khôi phục."""
    g = Path(goc)
    if not g.is_dir():
        return []
    out = []
    for d in g.iterdir():
        m = d / "meta.json"
        if not m.is_file():
            continue
        try:
            info = json.loads(m.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        n = sum(1 for _ in open(d / "events.jsonl", encoding="utf-8")) \
            if (d / "events.jsonl").is_file() else 0
        out.append({"session_id": d.name, "query": info.get("query_text", ""),
                    "task": info.get("task", ""), "n_events": n,
                    "created_at": info.get("created_at", 0)})
    return sorted(out, key=lambda x: -x["created_at"])
