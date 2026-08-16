"""
k-best Viterbi cho căn chỉnh chuỗi sự kiện TRAKE (D3)
=====================================================

ĐỘNG CƠ — Định lý 3 (TRAKE không tách được theo mốc).
    max_i (1/N)·Σⱼ I_{ij}  ≤  (1/N)·Σⱼ max_i I_{ij},  thường là bất đẳng thức ngặt.

    Nộp N câu trả lời, mỗi câu đúng đúng 1 mốc → Final ≈ 1/N.
    Nộp 1 câu trả lời đúng cả N mốc        → Final = 1.

    Vì `R@k` lấy max trên TỪNG CÂU TRẢ LỜI (mỗi câu là một bộ N khung hình),
    việc hệ (hedging) phải sinh ra nhiều BỘ N-tuple hoàn chỉnh, chứ không phải
    nhiều ứng viên rời rạc cho từng mốc. Đó chính là bài toán k-best path.

BÀI TOÁN.
    Cho N mốc, M khung hình ứng viên (tăng dần), ma trận điểm S[j][i].
    Tìm K bộ gán (i₀ < i₁ < … < i_{N−1}) khác nhau, thoả ràng buộc khoảng cách
    tối thiểu, có tổng điểm Σⱼ S[j][iⱼ] lớn nhất.

THUẬT TOÁN — List Viterbi song song (parallel list Viterbi).
    dp[j][i] = danh sách K đường tốt nhất kết thúc tại (mốc j, khung hình i).

    Tính đúng vì: đường tốt thứ k đi vào trạng thái (j, i) phải có phần đầu là
    một trong K đường tốt nhất của trạng thái tiền nhiệm — nếu không, thay phần
    đầu bằng đường tốt hơn sẽ cho đường tốt hơn mà vẫn kết thúc tại (j, i),
    mâu thuẫn. Hậu tố S[j][i] là hằng số nên không đổi thứ tự. ∎

    Duy trì "pool K tốt nhất trên tiền tố hợp lệ" khi `i` tăng dần ⟹ độ phức
    tạp O(N · M · K) thay vì O(N · M² · K) của cách quét lại toàn bộ tiền tố.

ĐIỂM CỘNG TÍNH. `S` phải là thang CỘNG TÍNH. Nếu bạn có xác suất, truyền
log-xác suất — khi đó tổng điểm là log của xác suất khớp toàn bộ chuỗi.

PHẠT NHỊP ĐỘ NẰM **TRONG** DP, không phải hậu kỳ (`pacing_penalty`).
    DP[j][i] = S[j][i] + max_{τ hợp lệ} ( DP[j−1][τ] − λ·(c[i] − c[τ]) )

    Tách hằng số ra khỏi phép `max` để giữ nguyên O(N·M·K):

        DP[j][i] = S[j][i] − λ·c[i] + max_τ ( DP[j−1][τ] + λ·c[τ] )

    ⟹ pool chỉ cần xếp hạng theo khoá `DP[j−1][τ] + λ·c[τ]`; phần `−λ·c[i]` là
    hằng số với mọi phần tử trong pool nên không đổi thứ hạng. Cùng độ phức tạp
    với λ = 0, và khi λ = 0 thì hai đường hoàn toàn trùng nhau.

    VÌ SAO KHÔNG ĐỂ HẬU KỲ. `apply_pacing_penalty` chỉ xếp lại `k` bộ ĐÃ được
    chọn — nó không cứu được bộ có nhịp độ tốt nhưng bị loại ngay ở vòng sinh
    k-best vì tổng điểm khớp-mốc thô thua. Đó là chế độ hỏng thật, không phải
    giả định: xem `TestPacingInsideDpVsPostHoc` trong `tests/test_kbest.py`,
    nơi đường đúng về nhịp độ nằm NGOÀI top-k thô nên hậu kỳ vĩnh viễn không
    nhìn thấy nó.

    CHỌN λ — CHÚ Ý ĐƠN VỊ, ĐÂY LÀ CHỖ DỄ SAI 20 LẦN.

    DANTE (AIO_Owlgorithms, arXiv:2512.13169, Eq. 2) công bố λ ∈ [0.001, 0.01]
    tinh chỉnh ở vòng chung kết AIC HCMC 2025: λ=0.001 hợp khi khoảng cách
    **chỉ số 3–15**, λ=0.01 hợp khi căn chỉnh chặt **1–3 chỉ số**.

    "Chỉ số" của họ là **chỉ số KEYFRAME** (`t` chạy trên [1,T] gồm T keyframe,
    mỗi video chiếm dải [s_v, e_v]), KHÔNG phải chỉ số khung hình. Hàm này nhận
    `candidates` là `frame_id`, nên `t − τ` tính bằng KHUNG HÌNH — phải chia lại:

        λ_khung = λ_chỉ_số / (số khung mỗi bước chỉ số keyframe)

    Đo trên bộ keyframe hiện tại: khoảng cách trung bình giữa hai keyframe liên
    tiếp là **81.3 khung** (trung vị 54). Vậy:

        λ ∈ [0.001, 0.01] / 81.3  =  **[1.2e-5, 1.2e-4]**

    Dùng thẳng 0.001–0.01 trên thang khung hình sẽ phạt **mạnh gấp ~81 lần** ý
    định của họ, ép mọi bộ co về sát nhau bất kể điểm khớp.

    Cách khác, sạch hơn về đơn vị: truyền `candidates` là chỉ số keyframe (0,1,
    2,…) rồi ánh xạ ngược ra `frame_id` sau — khi đó λ của họ dùng nguyên vẹn.

    Chưa grid-search trên dữ liệu thật — đây là điểm khởi đầu có căn cứ, không
    phải giá trị đã chốt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignedTuple:
    """Một bộ N khung hình hoàn chỉnh — tương ứng MỘT câu trả lời TRAKE."""

    frames: tuple[int, ...]
    score: float

    def __len__(self) -> int:
        return len(self.frames)


def k_best_alignments(
    candidates: Sequence[int],
    scores: Sequence[Sequence[float]],
    k: int,
    *,
    min_gap: int = 1,
    pacing_penalty: float = 0.0,
) -> list[AlignedTuple]:
    """
    Tìm `k` bộ căn chỉnh tốt nhất thoả ràng buộc thứ tự thời gian.

    Args:
        candidates: khung hình ứng viên, PHẢI tăng dần nghiêm ngặt.
        scores: ma trận N × M, `scores[j][i]` = điểm gán mốc `j` cho
            `candidates[i]`. Thang cộng tính (dùng log-xác suất nếu có).
        k: số bộ cần trả về.
        min_gap: khoảng cách khung hình tối thiểu giữa hai mốc liên tiếp.
            `min_gap=1` nghĩa là chỉ cần tăng nghiêm ngặt.
        pacing_penalty: `λ` — phạt mỗi khung hình khoảng cách giữa hai mốc liên
            tiếp, áp NGAY TRONG quy hoạch động (xem docstring module). `0.0` =
            tắt, cho kết quả TRÙNG KHỚP bản không phạt. Điểm khởi đầu gợi ý:
            0.00026–0.0026.

    Returns:
        Danh sách ≤ k `AlignedTuple`, sắp xếp giảm dần theo `score`. `score` đã
        TRỪ phần phạt nhịp độ — nó là giá trị mục tiêu mà DP tối ưu, không phải
        tổng điểm khớp-mốc thô. Rỗng nếu không tồn tại bộ nào thoả ràng buộc
        (ví dụ M quá nhỏ so với N).

    Raises:
        ValueError: candidates không tăng dần, kích thước scores không khớp,
            k < 1, hoặc pacing_penalty âm.
    """
    if k < 1:
        raise ValueError(f"k phải ≥ 1, nhận {k}")
    if min_gap < 1:
        raise ValueError(f"min_gap phải ≥ 1, nhận {min_gap}")
    if pacing_penalty < 0:
        raise ValueError(f"pacing_penalty không được âm, nhận {pacing_penalty}")

    m = len(candidates)
    n = len(scores)
    if n == 0 or m == 0:
        return []
    if any(len(row) != m for row in scores):
        raise ValueError(f"scores phải là ma trận {n}×{m}")
    if any(candidates[i] >= candidates[i + 1] for i in range(m - 1)):
        raise ValueError("candidates phải tăng dần nghiêm ngặt")

    # dp[i] cho mốc hiện tại: danh sách (score, prev_i, prev_rank), giảm dần.
    # prev_i = -1 nghĩa là mốc đầu tiên (không có tiền nhiệm).
    dp: list[list[tuple[float, int, int]]] = [[(scores[0][i], -1, -1)] for i in range(m)]
    # backpointers[j][i] = dp của mốc j tại khung hình i, giữ lại để truy vết.
    backpointers: list[list[list[tuple[float, int, int]]]] = [dp]

    for j in range(1, n):
        prev_dp = dp
        new_dp: list[list[tuple[float, int, int]]] = [[] for _ in range(m)]

        # `pool` = k đường tốt nhất trong số các tiền nhiệm HỢP LỆ đã duyệt.
        # Khi `i` tăng, tập tiền nhiệm hợp lệ chỉ nở ra, nên pool cập nhật tăng dần.
        #
        # KHOÁ XẾP HẠNG là `DP[j−1][p] + λ·c[p]`, KHÔNG phải `DP[j−1][p]`. Đây
        # chính là chỗ phạt nhịp độ đi vào DP: phần còn lại của công thức,
        # `−λ·c[i]`, là hằng số với mọi phần tử pool tại một `i` nên không đổi
        # thứ hạng — nhờ vậy vẫn chỉ cần giữ k phần tử, độ phức tạp không đổi.
        # Với λ = 0 khoá này suy biến về `DP[j−1][p]`, trùng bản không phạt.
        pool: list[tuple[float, int, int]] = []
        next_to_add = 0

        for i in range(m):
            limit = candidates[i] - min_gap
            # Nạp mọi tiền nhiệm p có candidates[p] ≤ limit mà chưa nạp.
            while next_to_add < m and candidates[next_to_add] <= limit:
                bonus = pacing_penalty * candidates[next_to_add]
                entries = [
                    (sc + bonus, next_to_add, rank)
                    for rank, (sc, _, _) in enumerate(prev_dp[next_to_add])
                ]
                if entries:
                    pool = sorted(pool + entries, key=lambda t: t[0], reverse=True)[:k]
                next_to_add += 1

            if pool:
                s_ji = scores[j][i] - pacing_penalty * candidates[i]
                new_dp[i] = [(s_ji + key, pi, prank) for key, pi, prank in pool]

        dp = new_dp
        backpointers.append(dp)

    # Gom k đường tốt nhất trên toàn bộ trạng thái cuối.
    finals: list[tuple[float, int, int]] = []
    for i in range(m):
        finals.extend((sc, i, rank) for rank, (sc, _, _) in enumerate(dp[i]))
    finals.sort(key=lambda t: t[0], reverse=True)
    finals = finals[:k]

    results: list[AlignedTuple] = []
    for total, last_i, last_rank in finals:
        frames: list[int] = []
        j, i, rank = n - 1, last_i, last_rank
        while j >= 0:
            frames.append(candidates[i])
            _, prev_i, prev_rank = backpointers[j][i][rank]
            j, i, rank = j - 1, prev_i, prev_rank
        frames.reverse()
        results.append(AlignedTuple(frames=tuple(frames), score=total))

    return results


def best_alignment(
    candidates: Sequence[int],
    scores: Sequence[Sequence[float]],
    *,
    min_gap: int = 1,
    pacing_penalty: float = 0.0,
) -> AlignedTuple | None:
    """Đường tốt nhất duy nhất (Viterbi cổ điển). Trả `None` nếu vô nghiệm."""
    result = k_best_alignments(
        candidates, scores, k=1, min_gap=min_gap, pacing_penalty=pacing_penalty
    )
    return result[0] if result else None


def apply_pacing_penalty(tuples: Sequence[AlignedTuple], decay_rate: float) -> list[AlignedTuple]:
    """
    Phạt các bộ có khoảng cách LỚN giữa các mốc liên tiếp — ưu tiên chuỗi sự
    kiện diễn ra với nhịp độ HỢP LÝ hơn là chuỗi rải rác toàn video dù điểm
    khớp từng mốc cao.

    Động lực: `k_best_alignments` chỉ ép buộc THỨ TỰ tăng dần (Định lý 3),
    không có ưu tiên nào về ĐỘ GẦN giữa các mốc. Một chuỗi 4 sự kiện xảy ra
    trong 5 giây và một chuỗi 4 sự kiện tương tự rải trên cả video 3 phút có
    thể nhận điểm khớp-mốc như nhau, dù trường hợp đầu hợp lý hơn nhiều với
    "chuỗi sự kiện có cấu trúc" mà PDF mô tả (vd chạy đà→giậm nhảy→bay qua
    xà→tiếp đất — bốn khoảnh khắc liền mạch của MỘT hành động).

    Công thức, tương đương phần thưởng nhân `exp(-decay_rate·Δ)` mỗi cặp mốc
    liên tiếp cách nhau `Δ` khung hình — cộng dồn dưới dạng log để khớp thang
    điểm cộng tính đã dùng xuyên suốt:

        score_mới = score_gốc − decay_rate · Σᵢ (frames[i+1] − frames[i])

    ⚠️ **DÙNG `k_best_alignments(..., pacing_penalty=λ)` THAY CHO HÀM NÀY.**

    Hàm này chỉ xếp lại `k` bộ ĐÃ được chọn. Bộ có nhịp độ tốt nhưng bị loại
    NGAY ở vòng sinh k-best (vì tổng điểm khớp-mốc thô thua) thì hậu kỳ vĩnh
    viễn không nhìn thấy — nó không nằm trong đầu vào. Với `pacing_penalty`,
    phạt tham gia vào chính phép `max` của DP nên đường đó được giữ lại từ đầu.
    Hai đường CHỈ cho cùng kết quả khi mọi bộ tối ưu-có-phạt tình cờ đã nằm sẵn
    trong top-k thô — không có gì bảo đảm điều đó.

    Giữ lại vì vẫn dùng được ở nơi đã có sẵn danh sách `AlignedTuple` từ nguồn
    khác (vd hợp nhất nhiều nguồn ứng viên), và vì nó thuần tuý nên test rẻ.

    Args:
        tuples: kết quả của `k_best_alignments`, thứ tự bất kỳ.
        decay_rate: hệ số phạt mỗi khung hình khoảng cách. `0.0` = tắt, trả
            về nguyên trạng (đã sắp lại theo `score` gốc để nhất quán).

    Returns:
        Danh sách đã tính lại `score` và sắp giảm dần.
    """
    if decay_rate < 0:
        raise ValueError(f"decay_rate không được âm, nhận {decay_rate}")

    def total_gap(frames: tuple[int, ...]) -> int:
        return sum(b - a for a, b in zip(frames, frames[1:]))

    rescored = [
        AlignedTuple(frames=t.frames, score=t.score - decay_rate * total_gap(t.frames))
        for t in tuples
    ]
    rescored.sort(key=lambda t: t.score, reverse=True)
    return rescored
