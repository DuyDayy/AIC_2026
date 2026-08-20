"""
Ghi file nộp bài + validator chặn lỗi câm (D7)
==============================================

VÌ SAO CẦN MODULE NÀY. Lỗi nguy hiểm nhất của cuộc thi này là lỗi KHÔNG có
triệu chứng: nộp `frame_ms` thay vì `frame_id`, nộp 1 khung hình cho TRAKE
thay vì N, lệch chỉ số frame một hằng số. Pipeline vẫn chạy, file vẫn ghi ra,
và điểm về 0. Validator ở đây biến mọi lỗi loại đó thành exception ngay lúc
ghi file.

MẪU FILE NỘP CHÍNH THỨC CHƯA CÓ. Vì vậy phần *bất biến* (validator) tách hẳn
khỏi phần *định dạng* (`SubmissionFormat`). Khi BTC công bố mẫu, chỉ cần thêm
một lớp format mới; validator không đổi.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from src.scoring.rscore import MAX_ANSWERS

logger = logging.getLogger(__name__)

TaskType = Literal["kis", "qa", "trake"]


class SubmissionError(ValueError):
    """Vi phạm bất biến của file nộp bài. Luôn phải dừng, không được nuốt."""


@dataclass(frozen=True)
class TaskSubmission:
    """Toàn bộ câu trả lời cho MỘT truy vấn, đã sắp theo thứ hạng."""

    task_id: str
    task_type: TaskType
    # Mỗi phần tử là một câu trả lời: (video_id, các frame_id, answer tuỳ chọn).
    answers: tuple[tuple[str, tuple[int, ...], str | None], ...]
    # Số mốc N mà đề bài yêu cầu (TRAKE). KIS/QA luôn = 1.
    n_moments: int = 1
    scores: tuple[float, ...] = field(default=())


def validate_task(
    submission: TaskSubmission,
    *,
    budget: int = MAX_ANSWERS,
    frame_bounds: Mapping[str, int] | None = None,
    valid_frames: Mapping[str, set[int]] | None = None,
    require_full_budget: bool = False,
) -> None:
    """
    Kiểm mọi bất biến của một truy vấn. Ném `SubmissionError` ngay lỗi đầu tiên.

    Args:
        submission: câu trả lời của một truy vấn.
        budget: `B` — số câu trả lời tối đa (PDF: 100).
        frame_bounds: `{video_id: nb_frames}`. Nếu có, kiểm `0 ≤ frame_id < nb_frames`.
            Đây là hàng rào duy nhất bắt được lỗi lệch chỉ số frame (Định lý 5).
        require_full_budget: bật khi nộp bài thật — mọi slot tới `B` đều có
            trọng số > 0 (Định lý 2), bỏ trống là vứt điểm.
    """
    tid = submission.task_id
    if not tid:
        raise SubmissionError("task_id rỗng")
    if submission.n_moments < 1:
        raise SubmissionError(f"[{tid}] n_moments phải ≥ 1, nhận {submission.n_moments}")
    if submission.task_type in ("kis", "qa") and submission.n_moments != 1:
        raise SubmissionError(f"[{tid}] {submission.task_type} phải có đúng 1 mốc")

    n_answers = len(submission.answers)
    if n_answers > budget:
        raise SubmissionError(f"[{tid}] nộp {n_answers} câu, vượt ngân sách {budget}")
    if n_answers == 0:
        raise SubmissionError(f"[{tid}] không có câu trả lời nào")
    if require_full_budget and n_answers < budget:
        raise SubmissionError(
            f"[{tid}] chỉ nộp {n_answers}/{budget} câu — mọi slot tới {budget} "
            f"đều có trọng số > 0 (Định lý 2), bỏ trống là vứt điểm"
        )

    seen: set[tuple[str, tuple[int, ...]]] = set()
    for rank, (video_id, frames, answer) in enumerate(submission.answers, start=1):
        where = f"[{tid}] câu #{rank}"

        if not video_id:
            raise SubmissionError(f"{where}: video_id rỗng")

        if len(frames) != submission.n_moments:
            raise SubmissionError(
                f"{where}: có {len(frames)} khung hình, đề yêu cầu đúng "
                f"{submission.n_moments} (TRAKE cần N frame_id trong MỘT câu trả lời)"
            )

        for f in frames:
            # bool là subclass của int — chặn riêng để không lọt giá trị vô nghĩa.
            if isinstance(f, bool) or not isinstance(f, int):
                raise SubmissionError(
                    f"{where}: frame_id phải là số nguyên, nhận {f!r} ({type(f).__name__}). "
                    f"Nhắc: luật thi chấm theo frame_id, KHÔNG phải millisecond."
                )
            # Cổng chống nhầm `n` (số thứ tự keyframe) với `frame_idx` (số khung thật).
            # `frame_bounds` KHÔNG bắt được lỗi này: `n` luôn nhỏ hơn số khung nên luôn
            # lọt. Đo được 0/173.426 khung có n == frame_idx, lệch trung vị 5.267 — nên
            # nhầm là sai MỌI câu, im lặng hoàn toàn. Truyền `valid_frames` để chặn.
            if valid_frames is not None:
                ok = valid_frames.get(video_id)
                if ok is not None and f not in ok:
                    raise SubmissionError(
                        f"{where}: frame_id {f} không phải frame_idx nào của {video_id!r}. "
                        f"Nhầm `n` (số thứ tự keyframe) với `frame_idx` (số khung thật)? "
                        f"Dùng FlatIndex.answer() thay vì ids[row]."
                    )
            if f < 0:
                raise SubmissionError(f"{where}: frame_id âm ({f})")
            if frame_bounds is not None:
                nb = frame_bounds.get(video_id)
                if nb is None:
                    raise SubmissionError(f"{where}: không biết số khung hình của {video_id!r}")
                if f >= nb:
                    raise SubmissionError(
                        f"{where}: frame_id {f} ≥ số khung hình {nb} của {video_id!r}"
                    )

        if submission.n_moments > 1 and list(frames) != sorted(frames):
            raise SubmissionError(
                f"{where}: các mốc phải tăng dần theo thời gian, nhận {frames}"
            )

        if submission.task_type == "qa" and not (answer or "").strip():
            raise SubmissionError(f"{where}: truy vấn Q&A bắt buộc có answer")

        key = (video_id, tuple(frames))
        if key in seen:
            raise SubmissionError(f"{where}: trùng lặp câu trả lời {key} — lãng phí slot")
        seen.add(key)

    if submission.scores:
        if len(submission.scores) != n_answers:
            raise SubmissionError(f"[{tid}] số điểm ({len(submission.scores)}) ≠ số câu trả lời")
        if list(submission.scores) != sorted(submission.scores, reverse=True):
            raise SubmissionError(
                f"[{tid}] chưa sắp giảm dần theo điểm — vi phạm Định lý 2, "
                f"làm mất điểm ở các mốc R@k nhỏ"
            )


def validate_all(
    submissions: Sequence[TaskSubmission],
    *,
    budget: int = MAX_ANSWERS,
    frame_bounds: Mapping[str, int] | None = None,
    valid_frames: Mapping[str, set[int]] | None = None,
    require_full_budget: bool = False,
    expected_task_ids: Iterable[str] | None = None,
) -> None:
    """Kiểm toàn bộ file nộp bài, gồm cả việc thiếu/thừa truy vấn."""
    seen_ids: set[str] = set()
    for sub in submissions:
        if sub.task_id in seen_ids:
            raise SubmissionError(f"task_id trùng lặp: {sub.task_id}")
        seen_ids.add(sub.task_id)
        validate_task(
            sub,
            budget=budget,
            frame_bounds=frame_bounds,
            require_full_budget=require_full_budget,
        )

    if expected_task_ids is not None:
        expected = set(expected_task_ids)
        missing = expected - seen_ids
        extra = seen_ids - expected
        if missing:
            raise SubmissionError(f"thiếu {len(missing)} truy vấn, ví dụ: {sorted(missing)[:5]}")
        if extra:
            raise SubmissionError(f"thừa {len(extra)} truy vấn lạ: {sorted(extra)[:5]}")


# ============================================================
# ĐỊNH DẠNG — thay được khi BTC công bố mẫu chính thức
# ============================================================


def _answer_payload(
    task_type: TaskType,
    rank: int,
    video_id: str,
    frames: tuple[int, ...],
    answer: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"rank": rank, "video_id": video_id}
    if task_type == "trake":
        payload["frame_ids"] = list(frames)
    else:
        payload["frame_id"] = frames[0]
    if task_type == "qa":
        payload["answer"] = answer
    return payload


def build_payload(submissions: Sequence[TaskSubmission]) -> dict[str, Any]:
    """
    Dựng cấu trúc JSON để ghi ra đĩa.

    ⚠️ Mẫu chính thức của BTC CHƯA CÓ. Cấu trúc dưới đây bám sát định dạng câu
    trả lời trong PDF mục 2.1 (`<video_id>, <frame_id>` / `…, <answer>` /
    `<video_id>, <frame_id₁>, …, <frame_idₙ>`). Khi có mẫu thật, chỉ sửa hàm
    này — validator ở trên không đổi.
    """
    return {
        "predictions": [
            {
                "task_id": sub.task_id,
                "task_type": sub.task_type,
                "results": [
                    _answer_payload(sub.task_type, rank, vid, frames, ans)
                    for rank, (vid, frames, ans) in enumerate(sub.answers, start=1)
                ],
            }
            for sub in submissions
        ]
    }


def write_submission(
    submissions: Sequence[TaskSubmission],
    output_path: str | Path,
    *,
    budget: int = MAX_ANSWERS,
    frame_bounds: Mapping[str, int] | None = None,
    valid_frames: Mapping[str, set[int]] | None = None,
    require_full_budget: bool = False,
    expected_task_ids: Iterable[str] | None = None,
    calibration_dir: str | Path | None = "data",
) -> Path:
    """
    Kiểm rồi ghi file nộp bài. KHÔNG BAO GIỜ ghi khi validator chưa xanh.

    CỔNG TIER 0. Trước khi ghi, bắt buộc phải có kết quả hiệu chỉnh quy ước
    `frame_id` (`data/frame_index_calibration.json`, sinh bởi
    `scripts/calibration/00_calibrate_frame_index.py`). Lý do đây là cổng CHẶN chứ
    không phải cảnh báo: lệch quy ước một hằng số làm MỌI câu trả lời sai
    trong khi bài nộp vẫn đúng định dạng và qua sạch validator — Định lý 5 đo
    được rằng lệch hệ thống kéo `final` 1.00 → 0.50 mà `best` vẫn 1.0, tức
    không có chỉ số nội bộ nào báo động. Mọi bất biến khác trong file này đều
    kiểm được từ chính bài nộp; riêng δ thì không, nên nó cần bằng chứng
    ngoài.

    Nếu δ ≠ 0, hàm CỘNG δ vào mọi `frame_id` ngay tại đây — một chỗ duy nhất,
    sau khi validator đã chạy trên hệ toạ độ nội bộ.

    Args:
        calibration_dir: thư mục chứa `frame_index_calibration.json`. Đặt
            `None` để bỏ qua cổng — CHỈ dùng trong test, không dùng khi nộp
            thật.

    Returns:
        Đường dẫn file đã ghi.

    Raises:
        SubmissionError: bất kỳ bất biến nào bị vi phạm.
        RuntimeError: chưa chạy hiệu chỉnh Tier 0.
    """
    validate_all(
        submissions,
        budget=budget,
        frame_bounds=frame_bounds,
        require_full_budget=require_full_budget,
        expected_task_ids=expected_task_ids,
    )

    if calibration_dir is not None:
        from src.ingestion.frame_index import require_calibration

        calib = require_calibration(calibration_dir)
        if calib.delta != 0:
            logger.warning(
                "Quy ước frame_id lệch δ=%+d (nguồn: %s, %d mẫu) — đang bù trừ "
                "cho toàn bộ bài nộp.", calib.delta, calib.method, calib.n_samples
            )
            # `replace` thay vì dựng lại bằng tay: giữ nguyên MỌI trường khác
            # (`scores`, và bất kỳ trường nào thêm sau này) — dựng tay sẽ âm
            # thầm đánh rơi chúng.
            submissions = [
                replace(
                    s,
                    answers=tuple(
                        (vid, tuple(calib.apply(f) for f in frames), ans)
                        for vid, frames, ans in s.answers
                    ),
                )
                for s in submissions
            ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_payload(submissions), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    total = sum(len(s.answers) for s in submissions)
    logger.info(
        "Đã ghi %s: %d truy vấn, %d câu trả lời (trung bình %.1f/truy vấn).",
        path,
        len(submissions),
        total,
        total / len(submissions) if submissions else 0.0,
    )
    return path


def from_ranked_answers(
    task_id: str,
    task_type: TaskType,
    ranked: Sequence[Any],
    *,
    n_moments: int,
    answers: Sequence[str] | None = None,
) -> TaskSubmission:
    """
    Chuyển đầu ra của `allocator.allocate_submission` thành `TaskSubmission`.

    Args:
        ranked: danh sách `RankedAnswer` (đã sắp giảm dần theo điểm).
        answers: chuỗi answer cho Q&A, cùng độ dài với `ranked`.
    """
    if answers is not None and len(answers) != len(ranked):
        raise SubmissionError(f"[{task_id}] số answer ≠ số câu trả lời")
    return TaskSubmission(
        task_id=task_id,
        task_type=task_type,
        answers=tuple(
            (r.video_id, tuple(r.frames), answers[i] if answers else None)
            for i, r in enumerate(ranked)
        ),
        n_moments=n_moments,
        scores=tuple(r.score for r in ranked),
    )


# ============================================================
# ĐỊNH DẠNG CHÍNH THỨC AIC 2026 — CSV mỗi truy vấn, đóng gói .zip
# ============================================================
#
# Nguồn: "Hướng dẫn nộp bài sơ tuyển". Mẫu đã công bố, nên phần `build_payload`
# (JSON tạm) ở trên KHÔNG còn là đường nộp — nó chỉ còn dùng cho chẩn đoán nội bộ.
#
# Toàn bộ mục này tồn tại vì cùng một lý do với validator ở đầu file: mọi lỗi định
# dạng ở đây đều là lỗi CÂM. File vẫn ghi ra, vẫn mở được bằng Notepad, vẫn trông
# đúng — và điểm về 0 vì bộ chấm tách cột ra khác thứ ta nghĩ.

import csv as _csv
import io as _io
import re as _re
import zipfile as _zipfile

#: Thể lệ: "Answer (Q&A) có độ dài tối đa 100 ký tự".
MAX_ANSWER_CHARS: int = 100

#: Thư mục BẮT BUỘC nằm trong file zip. Thể lệ in đậm: "PHẢI có thư mục submission
#: bên trong file zip · KHÔNG nén trực tiếp các file CSV".
SUBMISSION_DIRNAME: str = "submission"

#: "Khuyến cáo: Tên file zip chỉ nên bao gồm các ký tự chữ hoặc số".
_SAFE_ZIP_NAME = _re.compile(r"^[A-Za-z0-9_]+$")

#: Hậu tố quy định loại truy vấn, lấy từ TÊN FILE đề bài (`query-3-qa.txt` ⟹ qa).
SUFFIX_TO_TYPE: dict[str, TaskType] = {"kis": "kis", "qa": "qa", "trake": "trake"}


def task_type_from_filename(stem: str) -> TaskType:
    """
    `query-4-trake` → `"trake"`. Quy ước tên file là thứ DUY NHẤT nói loại truy vấn.

    Khớp bằng ĐUÔI, không bằng `in`: `"qa" in "query-1-kis"` là **True** (chữ `q`
    của `query`… không, nhưng `"kis" in "query-2-kis-qa-follow"` thì thật sự nhập
    nhằng). Thể lệ nói "hậu tố", nên ta đọc đúng hậu tố sau dấu `-` cuối cùng.
    """
    tail = stem.rsplit("-", 1)[-1].strip().lower()
    if tail not in SUFFIX_TO_TYPE:
        raise SubmissionError(
            f"tên file {stem!r} không kết thúc bằng hậu tố hợp lệ "
            f"({'/'.join(SUFFIX_TO_TYPE)}) — không xác định được loại truy vấn"
        )
    return SUFFIX_TO_TYPE[tail]


def clean_answer(answer: str) -> tuple[str, list[str]]:
    """
    Chuẩn đáp án Q&A về đúng thứ bộ chấm so khớp. Trả `(đáp án, các ghi chú)`.

    Ba phép, mỗi phép chữa một lỗi CÂM khác nhau:

    1. **Bỏ khoảng trắng đầu/cuối.** Thể lệ nói rõ hai điều cạnh nhau: "Khoảng trắng
       đầu/cuối: Được giữ nguyên, không tự động trim" và "Answer sẽ được so sánh dưới
       dạng chuỗi chính xác". Ghép lại: `" 5"` KHÁC `"5"` và bị chấm sai. Bộ chấm
       không trim hộ, nên ta phải tự trim.
    2. **Gộp xuống dòng thành khoảng trắng.** Xuống dòng trong ô là hợp lệ nếu có
       ngoặc kép, nhưng nó không bao giờ là đáp án đúng — nó là dấu hiệu ⑥ trả ra
       văn xuôi nhiều dòng.
    3. **Cắt còn 100 ký tự.** CẮT chứ không NÉM: ném lúc ghi file là mất TOÀN BỘ bài
       nộp vì một câu, trong khi cắt thì câu đó gần như chắc chắn đã sai sẵn (đáp án
       đúng của bộ đề này là "5", "Năm người", "Màu đỏ" — chạm 100 ký tự nghĩa là ⑥
       trả văn xuôi). Đổi một câu hỏng lấy 39 câu còn lại là đúng chiều.
       Luôn kèm ghi chú, không bao giờ im lặng.
    """
    notes: list[str] = []
    out = " ".join(answer.split())
    if out != answer:
        notes.append("đã bỏ khoảng trắng thừa/xuống dòng (bộ chấm KHÔNG tự trim)")
    if len(out) > MAX_ANSWER_CHARS:
        notes.append(f"đáp án {len(out)} ký tự > {MAX_ANSWER_CHARS} — ĐÃ CẮT")
        out = out[:MAX_ANSWER_CHARS]
    return out, notes


def format_task_csv(submission: TaskSubmission) -> tuple[str, list[str]]:
    """
    Dựng nội dung CSV của MỘT truy vấn theo thể lệ. Trả `(nội dung, ghi chú)`.

    Dùng `csv.writer` với `QUOTE_MINIMAL` chứ KHÔNG nối chuỗi bằng `","`. Đây không
    phải chuyện gọn mã: `QUOTE_MINIMAL` cài đúng nguyên văn ba luật của thể lệ —
    bọc ngoặc kép khi ô chứa dấu phẩy, khi chứa ngoặc kép (và nhân đôi ngoặc kép
    bên trong), khi chứa xuống dòng; không bọc trong các trường hợp còn lại. Nối
    tay bằng `","` thì đáp án `"Năm người, gồm nam và nữ"` biến thành HAI cột và cả
    dòng lệch — đúng một trong "5 lỗi thường gặp nhất" mà thể lệ liệt kê.

    `lineterminator="\\n"`: thể lệ nhận cả LF lẫn CRLF. Không header — thể lệ:
    "File CSV bắt đầu trực tiếp bằng dữ liệu".
    """
    notes: list[str] = []
    buf = _io.StringIO()
    w = _csv.writer(buf, lineterminator="\n")   # QUOTE_MINIMAL là mặc định
    for rank, (video_id, frames, answer) in enumerate(submission.answers, start=1):
        if video_id.lower().endswith(".mp4"):
            raise SubmissionError(
                f"[{submission.task_id}] câu #{rank}: tên video {video_id!r} còn đuôi "
                f".mp4 — thể lệ đòi 'L01_V028', không phải 'L01_V028.mp4'"
            )
        row: list[object] = [video_id, *(int(f) for f in frames)]
        if submission.task_type == "qa":
            a, n = clean_answer(answer or "")
            notes += [f"[{submission.task_id}] {m}" for m in n]
            if not a:
                raise SubmissionError(
                    f"[{submission.task_id}] câu #{rank}: đáp án Q&A rỗng sau khi "
                    f"chuẩn hoá — nộp thiếu cột answer là 0 điểm"
                )
            row.append(a)
        w.writerow(row)
    return buf.getvalue(), notes


def parse_task_csv(text: str, task_type: TaskType, n_moments: int) -> list[list[str]]:
    """
    Đọc NGƯỢC file CSV vừa ghi, bằng đúng `csv.reader` mà bộ chấm sẽ dùng.

    Đây là phép kiểm mạnh nhất trong file này, và là phép duy nhất kiểm được thứ ta
    thật sự quan tâm: **bộ chấm tách ra đúng số cột ta định không**. Mọi kiểm khác
    chạy trên cấu trúc trong bộ nhớ; kiểm này chạy trên chuỗi byte sẽ nộp đi.
    """
    want = 1 + n_moments + (1 if task_type == "qa" else 0)
    rows = [r for r in _csv.reader(_io.StringIO(text)) if r]
    for i, r in enumerate(rows, start=1):
        if len(r) != want:
            raise SubmissionError(
                f"dòng {i} tách ra {len(r)} cột, thể lệ đòi {want} "
                f"({task_type}, N={n_moments}): {r!r}"
            )
        if not r[0] or r[0] != r[0].strip():
            raise SubmissionError(f"dòng {i}: cột video {r[0]!r} rỗng hoặc dính khoảng trắng")
        for c in r[1:1 + n_moments]:
            if not c.lstrip("-").isdigit():
                raise SubmissionError(
                    f"dòng {i}: frame_id {c!r} không phải số nguyên thuần — "
                    f"thể lệ: 'Frame ID sẽ được so sánh dưới dạng số nguyên'"
                )
        if task_type == "qa" and len(r[-1]) > MAX_ANSWER_CHARS:
            raise SubmissionError(f"dòng {i}: đáp án {len(r[-1])} ký tự > {MAX_ANSWER_CHARS}")
    return rows


def write_task_csv(submission: TaskSubmission, out_dir: str | Path,
                   *, budget: int = MAX_ANSWERS) -> tuple[Path, list[str]]:
    """
    Ghi `<out_dir>/<task_id>.csv` theo thể lệ, rồi ĐỌC LẠI để tự kiểm.

    Tên file lấy thẳng `task_id`, vốn là `stem` của file đề (`query-1-kis.txt` ⟹
    `query-1-kis.csv`). Thể lệ: "Tên file khớp với tên truy vấn".
    """
    validate_task(submission, budget=budget)
    text, notes = format_task_csv(submission)
    parse_task_csv(text, submission.task_type, submission.n_moments)   # tự kiểm
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{submission.task_id}.csv"
    # `newline=""` để Python không đổi `\n` thành `\r\n` lần thứ hai trên Windows.
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return p, notes


def pack_submission_zip(csv_dir: str | Path, zip_path: str | Path,
                        *, expected: Iterable[str] | None = None) -> Path:
    """
    Nén các `.csv` thành `<zip>/submission/*.csv` đúng cấu trúc thể lệ đòi.

    Hai điều hàm này cố ý làm khác `zip -r`:

    · **Chỉ nhận `.csv`.** Thư mục đầu ra của `run.py` còn có `_report.json` và
      `_rerank_scores.npz` (hàng chục MB). `zip -r` gói cả hai: một file lạ trong
      `submission/` là rủi ro parse, còn npz thì biến bài nộp thành file khổng lồ
      không lý do.
    · **Tự tạo tiền tố `submission/`** thay vì trông chờ thư mục nguồn tên đúng.
      "Thiếu thư mục submission" là lỗi số 2 trong danh sách lỗi thường gặp của
      thể lệ, và nó chỉ lộ ra sau khi đã tiêu một lượt nộp — mà mỗi gói chỉ có 3.
    """
    d = Path(csv_dir)
    files = sorted(p for p in d.glob("*.csv") if p.is_file())
    if not files:
        raise SubmissionError(f"{d} không có file .csv nào — bài nộp rỗng")
    if expected is not None:
        want = set(expected)
        got = {p.stem for p in files}
        if want - got:
            raise SubmissionError(f"thiếu {sorted(want - got)}")
        if got - want:
            raise SubmissionError(f"thừa file lạ: {sorted(got - want)}")

    z = Path(zip_path)
    if not _SAFE_ZIP_NAME.match(z.stem):
        raise SubmissionError(
            f"tên zip {z.stem!r} có ký tự ngoài chữ/số/gạch dưới — thể lệ khuyến cáo "
            f"chỉ dùng chữ hoặc số"
        )
    z.parent.mkdir(parents=True, exist_ok=True)
    with _zipfile.ZipFile(z, "w", _zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=f"{SUBMISSION_DIRNAME}/{p.name}")

    with _zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        bad = [n for n in names if not n.startswith(f"{SUBMISSION_DIRNAME}/")]
        if bad:
            raise SubmissionError(f"file nằm ngoài {SUBMISSION_DIRNAME}/: {bad}")
    logger.info("Đã đóng gói %s: %d file CSV trong %s/", z, len(files), SUBMISSION_DIRNAME)
    return z
