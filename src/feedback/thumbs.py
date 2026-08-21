"""
Ảnh khung cho UI — đọc thẳng từ kho keyframe, KHÔNG giải mã video
=================================================================

Ba kho `.zip` trong `data/keyframes/` là ảnh của CHÍNH bộ khung đang chạy:

    L21-L25.zip   images/<video>/<n:06d>.webp      284.748 ảnh
    L26.zip       L26/<video>/<n:06d>.webp         214.044
    L27-L30.zip   <Lxx>/<video>/<n:06d>.webp       110.684
                                                   ────────
                                                   609.476  = đúng chỉ mục

Đánh số theo `n` của chỉ mục, KHÔNG phải số thứ tự chạy. [ĐO] `L21_V001` có 1.778
ảnh với file cuối là `003711.webp`, và `000004.webp` KHÔNG tồn tại — đúng chỗ chỉ mục
nhảy từ n=3 sang n=6. Đó là bằng chứng ánh xạ khớp, không phải suy đoán.

VÌ SAO KHÔNG GIẢI NÉN. Đọc một mục từ zip là seek + inflate một entry: [ĐO] **40 ảnh
trong 8 ms**, so với **1.290 ms** khi cắt từ video bằng ffmpeg — nhanh 160 lần, và
tốn 0 byte đĩa thêm cho 609.476 file.

VÌ SAO KHÔNG DÙNG ffmpeg NỮA. Bản trước cắt bằng `-vf select=eq(n\\,N)`, tức giải mã
TỪ KHUNG 0 tới khung N: [ĐO] 6,92 s cho một khung sâu, và UI xin 40 ảnh ⟹ 2–4 phút,
trông y hệt treo. `-ss` hạ xuống 0,07 s, nhưng kho ảnh vẫn nhanh hơn 20 lần nữa và
không phụ thuộc việc có file `.mp4` trên đĩa. ffmpeg giữ lại làm ĐƯỜNG LÙI.

An toàn luồng: MỘT handle mỗi zip, có khoá. Không dùng `threading.local`: mở handle
là đọc mục lục 285.000 mục, [ĐO] 0,67 s — với 8 luồng thì trả phí đó 8 lần và 40 ảnh
mất 6,6 s thay vì 8 ms. Đọc một mục chỉ tốn ~0,2 ms nên tranh khoá không đáng kể;
handle chung ĐÚNG hơn ở đây, không phải đánh đổi.
"""

from __future__ import annotations

import base64
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

__all__ = ["KhoAnh", "cat_bang_ffmpeg"]

#: `(tên file zip, tiền tố bên trong)`. Tiền tố `""` nghĩa là `<Lxx>/<video>/...`.
KHO = (("L21-L25.zip", "images/"), ("L26.zip", ""), ("L27-L30.zip", ""))


class KhoAnh:
    """Tra ảnh theo `(video_id, n)`. Dựng bảng tra MỘT lần lúc khởi động."""

    def __init__(self, goc: str | Path = "data/keyframes") -> None:
        self.goc = Path(goc)
        self._z_cache: dict[str, zipfile.ZipFile] = {}
        self._khoa = threading.Lock()
        self.bang: dict[tuple[str, int], tuple[str, str]] = {}
        for ten, _tien_to in KHO:
            p = self.goc / ten
            if not p.is_file():
                continue
            with zipfile.ZipFile(p) as z:
                for nm in z.namelist():
                    if not nm.endswith(".webp"):
                        continue
                    phan = nm.split("/")
                    if len(phan) < 2:
                        continue
                    vid, so = phan[-2], phan[-1][:-5]
                    if so.isdigit():
                        self.bang[(vid, int(so))] = (ten, nm)

    def __len__(self) -> int:
        return len(self.bang)

    def mo_san(self) -> None:
        """Mở sẵn mọi handle lúc khởi động, để lần đọc đầu không gánh 0,67 s."""
        for ten, _ in KHO:
            if (self.goc / ten).is_file():
                self._z_cache.setdefault(ten, zipfile.ZipFile(self.goc / ten))

    def doc(self, video_id: str, n: int) -> bytes:
        """Ảnh WebP thô. `b""` nếu không có — bên gọi tự quyết đường lùi."""
        vt = self.bang.get((str(video_id), int(n)))
        if vt is None:
            return b""
        try:
            with self._khoa:
                z = self._z_cache.get(vt[0])
                if z is None:
                    z = self._z_cache[vt[0]] = zipfile.ZipFile(self.goc / vt[0])
                return z.read(vt[1])
        except (KeyError, zipfile.BadZipFile, OSError):
            return b""

    def doc_nhieu(self, keys: list[tuple[str, int]], workers: int = 8) -> list[str]:
        """Nhiều ảnh song song → base64, giữ nguyên thứ tự."""
        if not keys:
            return []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            return list(ex.map(
                lambda k: base64.b64encode(self.doc(k[0], k[1])).decode(), keys))


def cat_bang_ffmpeg(video_id: str, pts_time: float, *,
                    video_root: str = "data/video", rong: int = 320) -> bytes:
    """
    ĐƯỜNG LÙI khi kho ảnh thiếu khung. `-ss` TRƯỚC `-i` để nhảy O(1).

    [ĐO] `-ss` cho ảnh TRÙNG BYTE với `select=eq(n,N)` ở cùng độ rộng — xem
    `tests/test_thumbs.py`. Không có `-ss` thì ffmpeg giải mã từ khung 0: 6,92 s
    một khung.
    """
    import subprocess

    src = Path(video_root) / f"{video_id}.mp4"
    if not src.is_file():
        return b""
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, float(pts_time)):.6f}",
         "-i", str(src), "-frames:v", "1", "-vf", f"scale={rong}:-2",
         "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "-"],
        capture_output=True)
    return r.stdout or b""
