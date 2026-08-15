"""
Chia việc cho N container song song — cân theo SỐ KHUNG, không theo số video
============================================================================

Gói Modal chặn **10 container đồng thời** (`BAO_CAO_THIET_KE.md` §11.2), nên thời
gian đồng hồ của cả lượt bằng thời gian của **container chậm nhất**. Chia việc lệch
thì 9 container xong sớm rồi ngồi chờ một container.

=============================================================================
VÌ SAO KHÔNG CHIA THEO SỐ VIDEO
=============================================================================

Số keyframe mỗi video lệch rất xa: đo trên bộ của team, video ít nhất **162** khung,
nhiều nhất **733** khung — chênh **4.5×**. Chia 873 video thành 10 phần bằng nhau về
SỐ VIDEO có thể cho phần nặng nhất gấp phần nhẹ nhất vài lần, và cái đó dịch trực
tiếp thành thời gian đồng hồ bị kéo dài.

Chia theo **tổng số khung** thì cân được. Bài toán là *makespan minimisation* trên
máy song song đồng nhất (P||Cmax) — NP-hard, nhưng thuật toán **LPT** (Longest
Processing Time first) có bảo đảm cổ điển của Graham (1969):

    Cmax(LPT) / Cmax(tối ưu)  ≤  4/3 − 1/(3m)

Với `m = 10` cận là **1.30**. Thực tế tốt hơn nhiều khi số việc lớn hơn số máy rất
nhiều (873 việc, 10 máy) — `balance_report` đo độ lệch thật thay vì tin cận.

=============================================================================
MỘT VIDEO KHÔNG BAO GIỜ BỊ CHIA ĐÔI
=============================================================================

Đơn vị chia là VIDEO, không phải khung. Lý do là khả năng chạy lại: đầu ra ghi một
file jsonl cho mỗi video, và bằng chứng "đã xong" duy nhất đáng tin là **file đọc
lại được** (cùng nguyên tắc `vector_index.pending_videos`). Nếu một video bị chia
cho hai container thì "xong" trở thành trạng thái phân tán, và container bị thu hồi
giữa lượt để lại một file khuyết mà không ai biết là khuyết.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Sequence

# Cận Graham (1969) cho LPT trên `m` máy đồng nhất.
def lpt_bound(m: int) -> float:
    """`Cmax(LPT)/Cmax(opt) ≤ 4/3 − 1/(3m)` — cận lý thuyết, để so với độ lệch đo được."""
    if m < 1:
        raise ValueError(f"m phải ≥ 1, nhận {m}")
    return 4 / 3 - 1 / (3 * m)


@dataclass(frozen=True)
class Shard:
    """Một phần việc giao cho một container."""

    index: int
    video_ids: tuple[str, ...]
    total_frames: int

    @property
    def n_videos(self) -> int:
        return len(self.video_ids)


def balance_shards(weights: Sequence[tuple[str, int]], n_shards: int) -> list[Shard]:
    """
    Chia `(video_id, số_khung)` thành `n_shards` phần cân theo tổng số khung.

    Thuật toán LPT: sắp việc GIẢM DẦN theo trọng số, rồi lần lượt gán mỗi việc vào
    phần đang nhẹ nhất (dùng heap nên `O(n log m)`).

    Sắp giảm dần là phần quan trọng: gán theo thứ tự tuỳ ý cũng "cân" nhưng mất bảo
    đảm — một video 733 khung đến cuối cùng sẽ dồn hết vào một phần đã đầy.

    Args:
        weights: `(video_id, số khung)`. Trọng số ≤ 0 bị TỪ CHỐI thay vì bỏ qua —
            một video 0 khung là dấu hiệu metadata hỏng, không phải việc nhẹ.
        n_shards: số container. Modal chặn 10 (§11.2).

    Returns:
        Đúng `n_shards` phần, sắp theo `index`. Phần có thể RỖNG nếu số video ít hơn
        số container — trả rỗng chứ không bỏ, để chỗ gọi luôn thấy đủ `n_shards`.

    Raises:
        ValueError: `n_shards < 1`, `video_id` trùng, hoặc trọng số ≤ 0.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards phải ≥ 1, nhận {n_shards}")
    ids = [v for v, _ in weights]
    if len(set(ids)) != len(ids):
        dup = sorted({v for v in ids if ids.count(v) > 1})
        raise ValueError(f"video_id trùng: {dup[:5]}")
    for v, w in weights:
        if w <= 0:
            raise ValueError(f"{v} có số khung {w} ≤ 0 — metadata hỏng, KHÔNG coi là việc nhẹ")

    # heap: (tổng khung hiện tại, chỉ số phần) — luôn lấy phần nhẹ nhất.
    heap = [(0, i) for i in range(n_shards)]
    heapq.heapify(heap)
    buckets: list[list[str]] = [[] for _ in range(n_shards)]
    totals = [0] * n_shards

    for video_id, w in sorted(weights, key=lambda kv: (-kv[1], kv[0])):
        load, i = heapq.heappop(heap)
        buckets[i].append(video_id)
        totals[i] = load + w
        heapq.heappush(heap, (totals[i], i))

    return [
        Shard(index=i, video_ids=tuple(buckets[i]), total_frames=totals[i])
        for i in range(n_shards)
    ]


def balance_report(shards: Sequence[Shard]) -> dict:
    """
    Số đo độ cân — thời gian đồng hồ tỉ lệ với `max_frames`, không với `total`.

    `imbalance` là `max/mean`: bằng 1.0 là cân hoàn hảo. So nó với `lpt_bound` để
    biết phép chia có đang bám cận lý thuyết hay không, thay vì tin cận.
    """
    if not shards:
        return {"n_shards": 0}
    totals = [s.total_frames for s in shards]
    mean = sum(totals) / len(totals)
    return {
        "n_shards": len(shards),
        "total_frames": sum(totals),
        "max_frames": max(totals),
        "min_frames": min(totals),
        "mean_frames": round(mean, 1),
        "imbalance": round(max(totals) / mean, 4) if mean else 0.0,
        "lpt_bound": round(lpt_bound(len(shards)), 4),
        "n_videos": [s.n_videos for s in shards],
    }


# =============================================================================
# QUYẾT ĐỊNH CPU HAY GPU — bằng số, không bằng cảm giác
# =============================================================================


def projected_cost(
    n_frames: int,
    frames_per_second: float,
    usd_per_hour: float,
    n_containers: int = 10,
) -> dict:
    """
    Chi phí và thời gian đồng hồ dự kiến cho một cấu hình.

    `frames_per_second` là throughput MỘT container đo được — phải đo, không suy.

    **Song song mua THỜI GIAN ĐỒNG HỒ, không mua chi phí.** Tổng thời gian container
    là `n_frames / fps` bất kể chạy trên 1 hay 10 container; chia 10 chỉ làm thời
    gian đồng hồ ngắn đi 10 lần, còn số container-giây phải trả thì như nhau. Nên
    `usd` KHÔNG phụ thuộc `n_containers`, và `n_containers` chỉ vào `wall_clock`.

    (Bỏ qua chi phí khởi động container. Với lượt hàng nghìn giây nó là nhiễu; với
    lượt ngắn thì không, và lúc đó `--benchmark` sẽ tự phơi ra vì nó đo end-to-end.)
    """
    if frames_per_second <= 0:
        raise ValueError("frames_per_second phải > 0")
    if n_containers < 1:
        raise ValueError(f"n_containers phải ≥ 1, nhận {n_containers}")
    container_seconds = n_frames / frames_per_second
    return {
        "container_seconds": round(container_seconds, 1),
        "wall_clock_seconds": round(container_seconds / n_containers, 1),
        "usd": round(container_seconds / 3600 * usd_per_hour, 4),
        "usd_per_1000_frames": round(1000 / frames_per_second / 3600 * usd_per_hour, 5),
    }


def choose_device(cpu: dict, gpu: dict, *, min_speedup: float = 3.0) -> tuple[str, str]:
    """
    Chọn CPU hay GPU từ HAI phép đo, theo quy tắc tường minh.

    Quy tắc, theo đúng yêu cầu "chỉ dùng GPU nếu hiệu quả đáng kể so chi phí":

      1. GPU **rẻ hơn** ⟹ chọn GPU. Không có đánh đổi nào phải cân.
      2. GPU đắt hơn nhưng nhanh hơn ≥ `min_speedup` lần ⟹ chọn GPU, và NÓI RÕ
         phần tiền trả thêm để mua thời gian.
      3. Còn lại ⟹ CPU.

    Args:
        cpu, gpu: kết quả `projected_cost` của hai cấu hình.

    Returns:
        `("cpu"|"gpu", lý do đọc được)` — lý do phải in ra, vì một lựa chọn hạ tầng
        không kèm con số là một lựa chọn không kiểm lại được.
    """
    c_usd, g_usd = cpu["usd"], gpu["usd"]
    c_wall, g_wall = cpu["wall_clock_seconds"], gpu["wall_clock_seconds"]
    speedup = c_wall / g_wall if g_wall > 0 else float("inf")

    if g_usd <= c_usd:
        return "gpu", (
            f"GPU RẺ HƠN: ${g_usd:.2f} so ${c_usd:.2f}, và nhanh hơn {speedup:.1f}× "
            f"({g_wall/60:.0f} phút so {c_wall/60:.0f} phút). Không có đánh đổi."
        )
    if speedup >= min_speedup:
        return "gpu", (
            f"GPU đắt hơn ${g_usd - c_usd:.2f} nhưng nhanh hơn {speedup:.1f}× "
            f"(≥ {min_speedup}× đã chốt): {g_wall/60:.0f} phút so {c_wall/60:.0f} phút. "
            f"Mua thời gian bằng ${g_usd - c_usd:.2f}."
        )
    return "cpu", (
        f"CPU: GPU đắt hơn ${g_usd - c_usd:.2f} mà chỉ nhanh hơn {speedup:.1f}× "
        f"(< {min_speedup}× đã chốt) ⟹ không đáng."
    )
