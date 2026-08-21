"""
Hoàn thiện bài nộp ĐÃ CÓ: sinh đáp án Q&A + ghi lại theo `FIELD_SEP`
====================================================================

KHÔNG chạy lại truy xuất. Đọc thẳng các `.csv` đã ghi, chỉ làm hai việc còn thiếu:

  ⑥ đáp án Q&A — Qwen2.5-VL trên Modal, ảnh cắt từ VIDEO GỐC theo `frame_idx`
  ⑦ ghi lại    — qua `write_task_csv` nên tự áp `FIELD_SEP`, cổng Tầng 0 và phép
                 đọc-ngược-tự-kiểm; rồi đóng gói `submission/*.csv` → `.zip`

Vì sao ảnh phải lấy từ video gốc: kho `aic-keyframes` đánh số theo `n` của bộ trích
xuất CŨ — [ĐO] 601 ảnh cho `L21_V001` trong khi chỉ mục hiện tại có 1.778 khung và
`n` chạy tới 3.711. Tra theo `n` ở đó cho ra ẢNH KHÁC, không có gì báo. Cắt thẳng
`frame_idx` khỏi `.mp4` thì chỉ có một cách hiểu — đã kiểm: cắt theo `frame_idx` và
cắt theo `pts_time` cho JPEG TRÙNG BYTE.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.submission.writer import (                                # noqa: E402
    FIELD_SEP, TaskSubmission, pack_submission_zip, task_type_from_filename,
    write_task_csv,
)
from run_thunghiem_modal import cat_khung                          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="submission_thunghiem/submission")
    ap.add_argument("--qdir", default="THUNGHIEM-bo-de-thi")
    ap.add_argument("--out", default="submission_final")
    ap.add_argument("--video-root", default="data/video")
    ap.add_argument("--no-qa", action="store_true")
    a = ap.parse_args()

    src = Path(a.sub)
    files = sorted(src.glob("*.csv"))
    print(f"đọc {len(files)} file từ {src}")

    subs = []
    for p in files:
        kind = task_type_from_filename(p.stem)
        rows = [r for r in csv.reader(open(p, encoding="utf-8"),
                                      skipinitialspace=True) if r]
        n_mom = len(rows[0]) - 1 - (1 if kind == "qa" else 0)
        ans = [(r[0], tuple(int(x) for x in r[1:1 + n_mom]),
                r[-1] if kind == "qa" else None) for r in rows]
        subs.append(TaskSubmission(p.stem, kind, tuple(ans), n_moments=n_mom))
    print("  " + " · ".join(f"{k} {sum(1 for s in subs if s.task_type == k)}"
                            for k in ("kis", "qa", "trake")))

    qa = [s for s in subs if s.task_type == "qa"]
    if qa and not a.no_qa:
        import modal

        # `n` của khung, để tra OCR — bài nộp chỉ có `frame_idx`.
        ids = np.load("data/embed/ids.npy", allow_pickle=True)
        FI = np.load("data/embed/frame_idx.npy")
        n_of = {(str(v), int(f)): int(n) for (v, n), f in zip(ids, FI)}

        can = {}
        for s in qa:
            v, fr, _ = s.answers[0]
            can[s.task_id] = (v, fr[0], n_of.get((v, fr[0])))
        print("\n⑥ khung sẽ đọc:")
        for k, (v, f, n) in can.items():
            print(f"   {k:<22} {v}@{f}  (n={n})")

        # ngữ cảnh chữ: quét MỘT lượt `ocr.jsonl`, chỉ giữ đúng các khoá cần
        need = {(v, n) for v, _f, n in can.values() if n is not None}
        ocr_of = {}
        with open("data/OCR/ocr.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                k = (d["video_id"], int(d["n"]))
                if k in need:
                    ocr_of[k] = d.get("text_ascii_folded") or ""
        print(f"   OCR tra được {len(ocr_of)}/{len(need)} khung")

        de = {p.stem: p.read_text(encoding="utf-8").strip()
              for p in Path(a.qdir).glob("*.txt")}
        jobs, ai = [], []
        for s in qa:
            v, f, n = can[s.task_id]
            b = cat_khung(v, f, a.video_root)
            if b is None:
                print(f"   ⚠ {s.task_id}: thiếu {v}.mp4 — giữ CHUA_SINH",
                      file=sys.stderr)
                continue
            jobs.append({"image_b64": base64.b64encode(b).decode(),
                         "question": de.get(s.task_id, ""),
                         "ocr": ocr_of.get((v, n), ""), "asr": ""})
            ai.append(s)
        if jobs:
            print(f"\n⑥ Qwen2.5-VL trên Modal: {len(jobs)} đề…", flush=True)
            fn = modal.Function.from_name("aic-query", "qa_answer")
            got = fn.remote(jobs)
            thay = {}
            for s, ans in zip(ai, got):
                thay[s.task_id] = ans
                print(f"   {s.task_id}: {ans!r}")
            subs = [
                TaskSubmission(s.task_id, s.task_type,
                               tuple((v, fr, thay[s.task_id]) for v, fr, _ in s.answers),
                               n_moments=s.n_moments)
                if s.task_id in thay else s
                for s in subs
            ]

    odir = Path(a.out) / "submission"
    ghi_chu = []
    for s in subs:
        _p, notes = write_task_csv(s, odir, budget=100)
        ghi_chu += notes
    z = pack_submission_zip(odir, Path(a.out) / "submission.zip",
                            expected=[s.task_id for s in subs])
    print(f"\n✓ {len(subs)} CSV → {z} ({z.stat().st_size / 1e3:.0f} KB) "
          f"· FIELD_SEP={FIELD_SEP!r}")
    for m in ghi_chu:
        print(f"  ⚠ {m}")


if __name__ == "__main__":
    main()
