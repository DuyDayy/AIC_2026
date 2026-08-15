"""
Tầng ① — PROBE HOÁ: đề thi (chuỗi) → danh sách probe để mã hoá
================================================================

**Probe** = một thứ đang tìm, sẽ thành vector để so với chỉ mục.

    đề (chuỗi)
      ├─ mấy MỐC?              1 (KIS/QA) hay N theo thứ tự (TRAKE)
      └─ chuỗi trong ngoặc kép → đường khớp CHÍNH XÁC ở ②, không qua embedding

Tầng này làm **đúng hai việc**, cả hai **xác định**: không model, không tham số, không
ngẫu nhiên. Cùng đầu vào cho cùng đầu ra, hôm nay và một năm sau.

Văn bản đưa cho encoder là **nguyên văn đoạn của mốc đó** — không rút gọn, không viết
lại. Hai ý tưởng làm thêm ở đây đều đã bị chính phép đo bác bỏ; xem cuối docstring.

=============================================================================
TÁCH MỐC — chỉ tách ở chỗ ĐÁNG TIN, đo trên 30 đề thật
=============================================================================

Từ nối trình tự **không phải** dấu hiệu đáng tin. Đo vị trí của chúng trong 30 đề:

    sau đó      đầu câu  5 · giữa câu 1     ⟹ tách được
    tiếp đó     đầu câu  2 · giữa câu 0     ⟹ tách được
    cuối cùng   đầu câu  2 · giữa câu 0     ⟹ tách được
    lần lượt    đầu câu  0 · giữa câu 2     ⟹ **KHÔNG BAO GIỜ** tách
    đầu tiên    đầu câu  0 · giữa câu 5     ⟹ **KHÔNG BAO GIỜ** tách

Hai từ cuối là bẫy thật, đọc nguyên văn thì rõ:

    q12 "các người mẫu **lần lượt** bước trên sàn"   ← một cảnh, không phải chuỗi
    q13 "2 mức giá … **lần lượt** là 9…"             ← liệt kê, không phải sự kiện

Cắt ở đó chẻ một cảnh liền mạch thành hai mốc rời rồi ép DP ở ④ đi tìm hai khoảnh khắc
cách nhau — sai hoàn toàn, và **không có gì báo**: DP vẫn chạy, vẫn trả điểm.

`N = len(probes)` đi thẳng vào ④, nên tách **thừa** nguy hiểm hơn tách **thiếu**. Quy
tắc vì vậy bảo thủ: mốc tường minh `E1:`/`E2:` trước (q10 và q22 dùng đúng khuôn đó),
rồi mới tới từ nối **ở đầu câu**, và không bao giờ tách giữa câu.

=============================================================================
HAI Ý TƯỞNG ĐÃ BỊ BÁC BỎ — giữ số đo để không ai dựng lại
=============================================================================

**1. Đa biến thể (mã hoá k cách diễn đạt rồi lấy max).** Đo trên 30 truy vấn có
ground truth:

    nguyên văn      R@1 8/30 · R@10 16/30 · trung vị hạng  8
    max biến thể    R@1 4/30 · R@10 13/30 · trung vị hạng 17     ← xấu hơn 17/30 câu
    mean biến thể   R@1 6/30 · R@10 14/30 · trung vị hạng 14

Lập luận sai của tôi là: *"max chỉ nâng, không hạ, nên không mất gì"*. Đúng về **điểm**,
sai về **hạng** — max nâng điểm khung đích, nhưng nâng điểm **mọi khung khác** nhiều
hơn, vì một câu ngắn chung chung (*"Nền phía sau tối."*) khớp mạnh với hàng nghìn khung.
Sàn nhiễu dâng nhanh hơn tín hiệu.

**2. Rút gọn truy vấn.** Thí nghiệm có kiểm soát: thêm câu dẫn kiểu đề thi (+32% độ
dài) vào 30 câu đã biết đáp án:

    chỉ mô tả hình      trung vị hạng  8
    + câu dẫn           trung vị hạng 11   ⟹ câu dẫn tốn 1,29×

Nhưng cắt sai còn hại hơn nhiều: bỏ hết trừ câu đầu cho trung vị **8 → 19 (2,4×)**. Và
8/30 câu **tốt lên** khi có câu dẫn — bối cảnh chủ đề đôi khi là dấu hiệu thật. Lợi ích
1,29× không đủ bù rủi ro cắt nhầm 2,4×.

Bằng chứng ban đầu cho ý tưởng này (cụm rút tay thắng nguyên văn 4/7 trên đề thi) nói
về **rút lõi thị giác**, không phải **cắt câu** — hai việc khác nhau, và tôi đã lấy
bằng chứng cho cái thứ nhất rồi cài cái thứ hai.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Từ nối CHỈ được tách khi đứng đầu câu. `lần lượt` và `đầu tiên` cố ý VẮNG MẶT.
BOUNDARY_WORDS = (
    "sau đó", "tiếp đó", "tiếp theo", "kế tiếp", "cuối cùng",
    "kết thúc", "sau khi", "rồi sau",
)

# Mốc tường minh `E1:` / `E2 :`. Đáng tin hơn mọi từ nối vì người ra đề tự đánh số.
EVENT_MARKER = re.compile(r"(?:^|\s)E\s*(\d+)\s*:", re.IGNORECASE)

# Kết câu. KHÔNG tách ở dấu phẩy — đề dùng phẩy để liệt kê thuộc tính của CÙNG một
# cảnh ("áo trắng, đội nón, quấn khăn rằn").
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Ngoặc kép mọi kiểu: thẳng, cong, và nháy đơn.
QUOTED = re.compile(r"[\"“”']([^\"“”']{2,80})[\"“”']")


@dataclass(frozen=True)
class Probe:
    """
    Một mốc cần tìm.

    `text` là **nguyên văn** đoạn của mốc đó — thứ tầng ② mã hoá. Không có trường biến
    thể: đa biến thể đã đo được là xấu hơn, xem docstring module.
    """

    index: int                      # thứ tự mốc, 0-based
    text: str
    quoted: tuple[str, ...] = ()


def sentences(text: str) -> list[str]:
    """Cắt câu theo `.!?`, KHÔNG theo dấu phẩy — xem `SENTENCE_END`."""
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]


def extract_quoted(text: str) -> tuple[str, ...]:
    """
    Chuỗi trong ngoặc kép, giữ nguyên văn và nguyên dấu, bỏ trùng.

    14,4% đề có chuỗi trích dẫn (*"PHÚ XUÂN – GIA ĐỊNH"*, *"Nà Ní"*). Chúng phải đi
    thẳng vào nguồn khớp chính xác ở ②: embedding làm mờ tên riêng, BM25 thì khớp đúng
    — đây là ca hiếm mà thưa thắng dày.

    Bỏ chuỗi dưới 2 ký tự: ngoặc quanh một ký tự thường là dấu nháy trong văn bản, không
    phải trích dẫn.
    """
    return tuple(dict.fromkeys(m.group(1).strip() for m in QUOTED.finditer(text)))


def split_events(query: str) -> list[str]:
    """
    `query` → danh sách đoạn, mỗi đoạn một mốc. Một mốc thì trả đúng một phần tử.

    Ưu tiên mốc tường minh `E1:`/`E2:`; nếu không có thì tách theo từ nối **ở đầu câu**.
    Không bao giờ tách giữa câu — xem docstring module.
    """
    q = " ".join(query.split())
    if not q:
        return []

    marks = list(EVENT_MARKER.finditer(q))
    if len(marks) >= 2:
        # Bỏ phần trước mốc đầu: đó là lời dẫn chung ("Cảnh lắp ráp trong một xưởng.")
        # chứ không phải một mốc.
        out = []
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(q)
            seg = q[m.end():end].strip()
            if seg:
                out.append(seg)
        if len(out) >= 2:
            return out

    segs, cur = [], []
    for s in sentences(q):
        low = s.lower()
        if any(low.startswith(w) for w in BOUNDARY_WORDS) and cur:
            segs.append(" ".join(cur))
            cur = [s]
        else:
            cur.append(s)
    if cur:
        segs.append(" ".join(cur))
    return segs or [q]


def build_probes(query: str) -> list[Probe]:
    """
    Đầu vào duy nhất của tầng ②. `query` → `[Probe]`, đã tách mốc và gắn trích dẫn.

    `N = len(probes)`: 1 cho KIS/QA, N cho TRAKE. Tầng ④ nhận đúng con số đó.

    Chuỗi trích dẫn gắn vào **mốc chứa nó**. Khi chỉ có một mốc thì mọi trích dẫn của
    cả câu đều thuộc về nó — không mất trích dẫn nào.
    """
    segs = split_events(query)
    if not segs:
        return []
    all_quoted = extract_quoted(query)
    probes = []
    for seg in segs:
        q = extract_quoted(seg) if len(segs) > 1 else all_quoted
        probes.append(Probe(index=len(probes), text=seg, quoted=q))
    return probes
