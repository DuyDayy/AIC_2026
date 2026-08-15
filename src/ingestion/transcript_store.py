"""
Tầng 3c — LỜI NÓI: gắn `frame_id` vào phụ đề để nối được với chỉ mục khung hình
================================================================================

Phụ đề tải từ YouTube (`data/subs/{video_id}.json`) chỉ có `start_ms`/`end_ms`.
Chỉ mục ảnh và chỉ mục OCR đều khoá theo `frame_id`. Module này bắc cầu giữa hai
hệ, để lời nói trở thành **tín hiệu theo khung hình** hợp nhất được trực tiếp
với hai tín hiệu kia thay vì chỉ hợp nhất được ở mức video.

QUY ĐỔI. `frame_id = ts_giây × fps`, đúng quy ước BTC (δ = 0) — đã kiểm chứng
hai lần độc lập: 177,321 dòng `map-keyframes` của BTC, và 173,426 dòng metadata
keyframe với `|frame_idx − pts_time × fps|` lớn nhất **0.0001**.

⚠️ **fps LẤY TỪ CHÍNH VIDEO ĐÓ, KHÔNG DÙNG HẰNG SỐ.** Collection có **bốn** giá
trị fps: 25.0 (550 video), 30.0 (61), 29.97003 (30), 26.438 (1). Dùng 25.0 cho
một video 30fps làm mọi `frame_id` trôi **20%** — ở phút thứ 10 là lệch 3,600
khung hình. Đây đúng loại lỗi Định lý 5 mô tả: bài nộp vẫn hợp lệ, mọi chỉ số
nội bộ vẫn đẹp, điểm bằng 0. `TranscriptStore` bắt buộc lấy fps qua
`KeyframeStore` nên không có đường nào truyền nhầm hằng số vào.

BIÊN NỬA MỞ, KHÔNG LÀM TRÒN HAI ĐẦU. Một đoạn phụ đề chiếm khoảng thời gian
thật `[start_ms, end_ms)`. Khung hình `f` nằm trong đoạn khi `f/fps×1000` thuộc
khoảng đó ⟹ `start_frame = ceil(start·fps)`, `end_frame = ceil(end·fps) − 1`.
Làm tròn cả hai đầu bằng `round` sẽ kéo vào một khung nằm TRƯỚC lúc câu nói bắt
đầu — sai nhỏ nhưng có hệ thống, và cộng dồn qua hàng trăm đoạn.

GIỚI HẠN PHẢI BIẾT: LỜI NÓI ≠ HÌNH ẢNH. Người dẫn có thể nói về thứ không hiện
trên màn hình. Đo thật trên 86 video (quét offset −40s→+40s, đối chiếu từ hiếm
giữa phụ đề và OCR): tương quan đạt đỉnh **2.7× nền**, trọng tâm **+2.3 giây**,
nhưng bề rộng nửa đỉnh tới **26 giây** (−8s → +18s). Nghĩa là mốc thời gian
định vị được ĐOẠN, không định vị được KHUNG. Vì vậy `window_ms` là tham số
TƯỜNG MINH: mặc định 0 (chỉ đúng khoảng thời gian câu nói), caller nào cần khớp
ngữ nghĩa phải tự nới ra và tự chịu trách nhiệm về độ rộng đó.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

FAILED_FILENAME = "_failed.json"


@dataclass(frozen=True)
class TranscriptSegment:
    """
    Một đoạn phụ đề, đã gắn khoảng khung hình theo quy ước BTC.

    `start_frame`/`end_frame` là biên **bao gồm cả hai đầu** của tập khung hình
    nằm trong đoạn. Khi đoạn ngắn hơn một chu kỳ khung hình, `end_frame` có thể
    nhỏ hơn `start_frame` — nghĩa là đoạn không chứa trọn khung nào; đó là sự
    thật cần giữ, không phải lỗi cần che.
    """

    video_id: str
    start_ms: float
    end_ms: float
    start_frame: int
    end_frame: int
    fps: float
    text: str
    text_folded: str

    @property
    def covers_no_frame(self) -> bool:
        return self.end_frame < self.start_frame

    def contains_frame(self, frame_id: int, window_ms: float = 0.0) -> bool:
        """Khung hình này có nằm trong đoạn (đã nới `window_ms` mỗi phía) không."""
        if window_ms <= 0:
            return self.start_frame <= frame_id <= self.end_frame
        margin = window_ms / 1000.0 * self.fps
        return (self.start_frame - margin) <= frame_id <= (self.end_frame + margin)


# Cài đặt DUY NHẤT nằm ở `src/text/fold.py`. Tái xuất ở đây để mã cũ khỏi phải sửa,
# nhưng KHÔNG cài lại — hai cài đặt lệch nhau một ký tự là điểm tụt mà không gì báo.
from src.text.fold import strip_diacritics  # noqa: F401


def ms_to_frame_id(ts_ms: float, fps: float) -> int:
    """
    Quy đổi mốc thời gian sang `frame_id` theo quy ước BTC (δ = 0).

    Thuần tuý. Đây là MỘT chỗ duy nhất trong đường ống thực hiện phép quy đổi
    này, để nó test được và để không ai viết lại `int(ts/1000*25)` ở chỗ khác.

    Raises:
        ValueError: `fps` không dương — không có giá trị mặc định nào đúng, và
            đoán một giá trị là cách chắc chắn nhất để sai không triệu chứng.
    """
    if fps <= 0:
        raise ValueError(f"fps phải > 0, nhận {fps}")
    return int(round(ts_ms / 1000.0 * fps))


def attach_frame_ids(
    segments: Iterable[dict], video_id: str, fps: float
) -> list[TranscriptSegment]:
    """
    Gắn khoảng `[start_frame, end_frame]` vào từng đoạn phụ đề.

    Thuần tuý (nhận `dict`, không đọc file) nên test được không cần `data/subs`.

    Args:
        segments: các đoạn `{"start_ms", "end_ms", "text"}` do
            `01_fetch_subtitles.py` ghi ra.
        video_id: để thông báo lỗi chỉ đúng chỗ.
        fps: fps CỦA CHÍNH video đó — xem cảnh báo ở docstring module.

    Raises:
        ValueError: `fps` không dương, thiếu trường bắt buộc, hoặc `end_ms`
            đứng trước `start_ms` (phụ đề hỏng — không sửa ngầm).
    """
    if fps <= 0:
        raise ValueError(f"[{video_id}] fps phải > 0, nhận {fps}")

    out: list[TranscriptSegment] = []
    for i, seg in enumerate(segments, start=1):
        for key in ("start_ms", "end_ms"):
            if key not in seg:
                raise ValueError(f"[{video_id}] đoạn {i} thiếu {key!r}")
        start_ms = float(seg["start_ms"])
        end_ms = float(seg["end_ms"])
        if end_ms < start_ms:
            raise ValueError(
                f"[{video_id}] đoạn {i} có end_ms={end_ms} < start_ms={start_ms}"
            )
        text = str(seg.get("text") or "")
        # Biên nửa mở [start, end) — xem docstring module.
        start_frame = math.ceil(start_ms / 1000.0 * fps)
        end_frame = math.ceil(end_ms / 1000.0 * fps) - 1
        out.append(
            TranscriptSegment(
                video_id=video_id,
                start_ms=start_ms,
                end_ms=end_ms,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                text=text,
                text_folded=strip_diacritics(text),
            )
        )
    return out


def read_transcript_json(path: str | Path) -> dict:
    """Đọc một file phụ đề. Tách khỏi phần thuần tuý ở trên."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frame_text_index(
    segments: Sequence[TranscriptSegment],
    frame_ids: Iterable[int],
    *,
    window_ms: float = 0.0,
) -> dict[int, str]:
    """
    `{frame_id: lời nói quanh khung hình đó}` — dạng dùng được cho BM25.

    Đây là hàm QUAN TRỌNG NHẤT của module: nó biến lời nói (liên tục theo thời
    gian) thành tín hiệu rời rạc theo khung hình, cùng khoá với chỉ mục ảnh và
    chỉ mục OCR. Chỉ sau bước này ba tín hiệu mới hợp nhất được ở mức khung hình.

    Khung hình không có lời nói nào phủ thì KHÔNG xuất hiện trong kết quả — mục
    rỗng chỉ làm loãng thống kê IDF mà không bao giờ khớp truy vấn nào.

    Args:
        segments: đầu ra của `attach_frame_ids`, thứ tự bất kỳ.
        frame_ids: các khung hình cần tra (thường là `frame_id` của keyframe).
        window_ms: nới mỗi phía bấy nhiêu mili-giây trước khi xét phủ. Mặc định
            0 = đúng khoảng thời gian câu nói. Xem giới hạn "lời nói ≠ hình ảnh"
            ở docstring module trước khi nới.
    """
    if not segments:
        return {}
    ordered = sorted(segments, key=lambda s: s.start_frame)
    starts = [s.start_frame for s in ordered]
    fps = ordered[0].fps
    margin = window_ms / 1000.0 * fps

    # `end_frame` không đơn điệu theo `start_frame` (đoạn sau có thể kết thúc
    # sớm hơn đoạn trước), nên không nhị phân tìm được ở đầu phải. Dùng hậu tố
    # max để biết "từ vị trí i trở đi còn đoạn nào vươn tới khung này không".
    max_end_from = [0] * (len(ordered) + 1)
    max_end_from[len(ordered)] = -(10**18)
    for i in range(len(ordered) - 1, -1, -1):
        max_end_from[i] = max(ordered[i].end_frame, max_end_from[i + 1])

    out: dict[int, str] = {}
    for fid in frame_ids:
        hi = bisect.bisect_right(starts, fid + margin)
        parts = [
            ordered[i].text
            for i in range(hi)
            if ordered[i].contains_frame(fid, window_ms) and ordered[i].text.strip()
        ]
        if parts:
            out[fid] = " ".join(parts)
    return out


# Nguồn ASR chuẩn: MỘT cây, không chia theo engine, xếp giống `data/Framme/` —
#
#     data/ASR/{L21-L25 | L26 | L27-L30}/results/{video_id}.json     873 file
#
# Bản trên đĩa tự mang xuất xứ ở khoá `source` ("youtube" 819 · "phowhisper" 54),
# nên không cần logic ưu tiên lúc đọc nữa.
#
# Vì sao là hằng số chứ không để mỗi script tự gõ: bố cục đã đổi BA lần
# (`data/subs/` phẳng → `data/ASR/{engine}/{nhóm}/results/` → `data/ASR/{nhóm}/
# results/`) và lần nào cũng có script bị bỏ lại phía sau, trả về RỖNG mà không
# báo lỗi. Một nguồn sự thật thì lần đổi sau chỉ sửa một chỗ.
#
# Bản phiên âm whisper-small cũ của 54 video KHÔNG bị xoá — nó nằm trong chính
# file đó dưới khoá `alternates["whisper-small"]`. Muốn lùi về nó thì đổi khoá
# lúc đọc, không phải khôi phục thư mục từ git. Xem `scripts/asr/07_consolidate.py`.
#
# ⚠️ `alternates` KHÔNG được trộn vào `segments`. Hai bản phiên âm của cùng một
# video là hai giả thuyết loại trừ nhau, không phải hai nguồn bổ sung — xem lý
# do đo được ở docstring `__init__` (đồng thuận Jaccard chỉ 67.9%).
ASR_DIRS: tuple[str, ...] = ("data/ASR",)

# VÌ SAO PhoWhisper THẮNG whisper-small, tức vì sao `source` của 54 video kia là
# "phowhisper". Đo trên cùng 54 video:
#
#                       whisper-small   PhoWhisper-medium
#     độ dài đoạn MAX      29,000 ms        14,816 ms
#     đoạn TRÒN GIÂY          58.1%             0.8%     ⟸ mốc thật, không lượng tử
#     ảo giác lặp        'Cảm ơn các bạn đã     không còn
#                         đón xem video này' ×9
#     no_speech                   0               11     ⟸ nhận đúng video không lời
#
# Điểm mấu chốt là video KHÔNG CÓ LỜI (múa lân, nhạc nền): PhoWhisper ghi
# `no_speech: true` với 0 đoạn — một khẳng định đúng — trong khi whisper-small
# lấp chỗ trống bằng chính những câu ảo giác trên. "Không có lời" và "chưa xử lý"
# là hai trạng thái khác nhau, và chỉ bản mới phân biệt được.
# Xem `BAO_CAO_THIET_KE.md` §8.6.
#
# Hai tập RỜI NHAU (819 + 54, 0 chồng lấn) nên bước gộp là phép HỢP, không phải
# phép chọn — không video nào có hai ứng viên cho khoá `source`.


def iter_transcript_files(root: str | Path) -> Iterator[Path]:
    """
    Mọi file phụ đề dưới `root`, quét CẢ HAI bố cục.

      * phẳng          — `{root}/*.json`             (bố cục cũ `data/subs/`)
      * theo nhóm      — `{root}/{nhóm}/results/*.json`  (bố cục `data/ASR/`)

    `_`-tiền tố bị bỏ ở MỌI cấp: `_superseded/`, `_quarantine/`, `_failed.json`,
    `_unreliable.json` là tạo tác phụ trợ, không phải phụ đề.

    Công khai vì các script dựng bộ đánh giá cần đúng danh sách này mà không cần
    dựng cả `TranscriptStore` (vốn đòi `KeyframeStore` để biết fps). Trước đây
    chúng tự glob `{dir}/L*.json` — chỉ đúng ở bố cục phẳng, và khi dữ liệu
    chuyển sang `data/ASR/` thì trả về RỖNG mà không có lỗi nào.
    """
    root = Path(root)
    if not root.is_dir():
        return
    yield from (p for p in sorted(root.glob("*.json")) if not p.name.startswith("_"))
    for p in sorted(root.glob("*/results/*.json")):
        if not p.name.startswith("_") and not p.parts[-3].startswith("_"):
            yield p


class TranscriptStore:
    """
    Cổng truy cập phụ đề, song song với `KeyframeStore` (ảnh) và `OcrStore` (chữ).

    fps LUÔN lấy từ `KeyframeStore` — không có tham số nào cho phép truyền hằng
    số vào. Đó là ràng buộc CẤU TRÚC, không phải quy ước: lệch fps là lỗi 0 điểm
    không triệu chứng, nên đường duy nhất lấy được fps phải là đường đúng.
    """

    def __init__(self, subs_dir: str | Path | Sequence[str | Path], keyframe_store):
        """
        `subs_dir` nhận MỘT thư mục hoặc NHIỀU, xếp theo THỨ TỰ ƯU TIÊN GIẢM DẦN.

        Vì sao cần nhiều nguồn: ASR đến từ hai nơi có đặc tính lỗi khác nhau —
        phụ đề YouTube (819 video) và Whisper (54 video YouTube không có). Bản
        trước chỉ đọc được một thư mục, nên 54 video kia VÔ HÌNH với toàn hệ dù
        độ phủ ghi là 873/873. Đó là hỏng im lặng: mọi phép đo thiếu 6%
        collection ở đúng nhóm HTV Sports/Entertainment — phần dữ liệu TRAKE
        quý nhất — mà không có lỗi nào báo.

        KHI TRÙNG, nguồn ĐẦU tiên thắng. Không gộp hai bản phiên âm của cùng một
        video: chúng là hai giả thuyết cạnh tranh về cùng đoạn âm thanh, gộp lại
        cho ra văn bản không ai từng nói. Đo trên 75 cặp có cả hai nguồn:
        đồng thuận Jaccard chỉ 67.9%, trung vị Whisper giữ được 80.8% từ của
        YouTube — quá xa để coi là bổ sung cho nhau.

        Chấp nhận cả bố cục phẳng (`{dir}/*.json`) lẫn bố cục theo nhóm giống
        `data/Framme/` (`{dir}/{nhóm}/results/*.json`).
        """
        dirs = ([subs_dir] if isinstance(subs_dir, (str, Path))
                else [Path(d) for d in subs_dir])
        self._dirs = [Path(d) for d in dirs]
        self._dir = self._dirs[0]          # tương thích ngược: gốc của `_failed.json`
        self._kf = keyframe_store

        self._paths: dict[str, Path] = {}
        self._source_of: dict[str, Path] = {}
        for root in self._dirs:
            for p in self._iter_transcripts(root):
                if p.stem not in self._paths:      # nguồn ĐẦU tiên thắng
                    self._paths[p.stem] = p
                    self._source_of[p.stem] = root
        self._cache: dict[str, list[TranscriptSegment]] = {}

    @staticmethod
    def _iter_transcripts(root: Path):
        """Xem `iter_transcript_files` — giữ tên cũ cho các chỗ gọi sẵn có."""
        yield from iter_transcript_files(root)

    def video_ids(self) -> list[str]:
        return sorted(self._paths)

    def source_of(self, video_id: str) -> Path:
        """Thư mục nguồn của một video — cần khi hai nguồn có đặc tính lỗi khác nhau."""
        return self._source_of[video_id]

    def path_of(self, video_id: str) -> Path:
        """
        Đường dẫn file phụ đề THẬT đang được dùng.

        Cần vì bố cục có hai dạng (phẳng và `{nhóm}/results/`): mọi thao tác trên
        file — cách ly, xoá, đối chiếu — phải hỏi store chứ đừng tự ghép
        `{dir}/{video_id}.json`, kiểu ghép đó chỉ đúng ở bố cục phẳng và hỏng IM
        LẶNG ở bố cục kia (file không tồn tại ⟹ bỏ qua, không lỗi).
        """
        if video_id not in self._paths:
            raise KeyError(f"Không có phụ đề cho video {video_id!r}")
        return self._paths[video_id]

    def has(self, video_id: str) -> bool:
        return video_id in self._paths

    def fps_of(self, video_id: str) -> float:
        refs = self._kf.keyframes(video_id)
        if not refs:
            raise ValueError(f"[{video_id}] không có keyframe ⟹ không biết fps")
        return refs[0].fps

    def segments(self, video_id: str) -> list[TranscriptSegment]:
        """Phụ đề của một video, đã gắn `frame_id` (có cache trong phiên)."""
        if video_id not in self._cache:
            if video_id not in self._paths:
                raise KeyError(f"Không có phụ đề cho video {video_id!r}")
            data = read_transcript_json(self._paths[video_id])
            self._cache[video_id] = attach_frame_ids(
                data.get("segments") or [], video_id, self.fps_of(video_id)
            )
        return self._cache[video_id]

    def frame_text_index(self, video_id: str, *, window_ms: float = 0.0) -> dict[int, str]:
        """`{frame_id: lời nói}` cho ĐÚNG các keyframe của video này."""
        ids = [r.frame_id for r in self._kf.keyframes(video_id)]
        return frame_text_index(self.segments(video_id), ids, window_ms=window_ms)

    def segments_for_frame(
        self, video_id: str, frame_id: int, *, window_ms: float = 0.0
    ) -> list[TranscriptSegment]:
        return [s for s in self.segments(video_id) if s.contains_frame(frame_id, window_ms)]

    def videos_without_transcript(self, all_video_ids: Iterable[str]) -> list[str]:
        return sorted(set(all_video_ids) - set(self._paths))

    def permanently_failed(self) -> dict[str, str]:
        """
        Video đã xác định KHÔNG có phụ đề YouTube — `01_fetch_subtitles.py` ghi.

        Tìm ở MỌI thư mục nguồn, và chấp nhận cả hai tên: `_failed.json` (bố cục
        phẳng cũ) lẫn `_no_vietnamese_track.json` (bố cục theo nhóm). Tên sau nói
        đúng bản chất hơn — đo trên 3 ca ngẫu nhiên: 0 track ở MỌI ngôn ngữ, tức
        kênh không bật phụ đề tự động, không phải lỗi tải.
        """
        out: dict[str, str] = {}
        for root in self._dirs:
            for name in (FAILED_FILENAME, "_no_vietnamese_track.json"):
                p = root / name
                if p.exists():
                    out.update(json.loads(p.read_text(encoding="utf-8")))
        return out

    def check_duration(self, video_id: str, *, tolerance_sec: float = 2.0) -> float:
        """
        Lệch (giây) giữa thời lượng phụ đề khai báo và thời lượng mp4 thật.

        Đây là phép kiểm ĐỒNG BỘ rẻ nhất và bắt được lỗi đắt nhất: nếu bản
        YouTube tải phụ đề về là một BẢN DỰNG KHÁC với mp4 thi đấu, mọi mốc
        thời gian lệch mà không có triệu chứng nào khác. Đo thật: 85/86 video
        khớp trong ±0.6s; riêng `L21_V022` lệch **53.5s** — đúng một bản dựng
        khác, đã phải tải lại.

        Returns:
            `mp4_giây − phụ_đề_giây`. Dương nghĩa là mp4 dài hơn.
        """
        data = read_transcript_json(self._paths[video_id])
        sub_sec = float(data.get("duration_sec") or 0.0)
        refs = self._kf.keyframes(video_id)
        mp4_sec = max(r.shot_end_frame for r in refs) / self.fps_of(video_id)
        delta = mp4_sec - sub_sec
        if abs(delta) > tolerance_sec:
            logger.warning(
                f"[{video_id}] thời lượng phụ đề ({sub_sec:.1f}s) lệch mp4 "
                f"({mp4_sec:.1f}s) tới {delta:+.1f}s — nghi phụ đề thuộc BẢN DỰNG "
                f"KHÁC. Mọi mốc thời gian của video này không đáng tin."
            )
        return delta

    def iter_all(self) -> Iterator[TranscriptSegment]:
        for vid in self.video_ids():
            yield from self.segments(vid)
