"""
BM25 trên văn bản BỎ DẤU — tầng truy xuất chạy được KHÔNG cần GPU
===================================================================

VÌ SAO TỰ CÀI. `rank_bm25` không nằm trong `requirements.txt`, và BM25 Okapi chỉ
là ~40 dòng có công thức chuẩn công bố. Thêm một dep để lấy 40 dòng, ở một dự án
sắp phải chạy offline lúc thi, là đánh đổi sai.

CÔNG THỨC (Robertson & Zaragoza 2009, "The Probabilistic Relevance Framework:
BM25 and Beyond", §3.1):

    score(q, d) = Σ_{t ∈ q}  IDF(t) · [ f(t,d)·(k₁+1) ] / [ f(t,d) + k₁·(1 − b + b·|d|/avgdl) ]

    IDF(t) = ln( 1 + (N − n(t) + 0.5) / (n(t) + 0.5) )

`+1` trong `ln` là biến thể KHÔNG ÂM: dạng gốc `ln((N−n+0.5)/(n+0.5))` ra ÂM khi
từ xuất hiện ở hơn nửa số tài liệu, khiến một từ phổ biến kéo TỤT điểm của tài
liệu chứa nó — vô lý và gây lỗi xếp hạng thật. Dạng `1 + …` chặn dưới ở 0.

VÌ SAO BỎ DẤU. Truy vấn tiếng Việt gõ không dấu là chuyện thường, và phụ đề tự
động lẫn OCR đều sai dấu có hệ thống. `text_folded` của `TranscriptStore` và
`frame_text_folded` của OCR đã bỏ dấu sẵn theo CÙNG một quy ước — dùng thẳng,
không chuẩn hoá lại (tránh nguồn sự thật thứ hai).

GIỚI HẠN ĐÃ BIẾT: tách từ theo khoảng trắng, nên "Hồ Chí Minh" thành 3 token rời.
Tách từ tiếng Việt đúng (coccoc-tokenizer) là nâng cấp đã ghi trong
`CHON_CONG_NGHE.md`; ở đây là đường CƠ SỞ để đo xem nâng cấp đó đáng bao nhiêu.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

# Mặc định của Robertson & Zaragoza. `k₁` điều tiết bão hoà tần suất, `b` điều
# tiết mức chuẩn hoá theo độ dài tài liệu.
DEFAULT_K1: float = 1.5
DEFAULT_B: float = 0.75

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Chữ cái tiếng Việt CÓ DẤU — dùng để phát hiện văn bản chưa bỏ dấu lọt vào.
_DIACRITIC_RE = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩị"
    r"òóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]"
)


def looks_unfolded(text: str) -> bool:
    """`True` nếu `text` còn dấu tiếng Việt — tức CHƯA đi qua `strip_diacritics`."""
    return bool(_DIACRITIC_RE.search(text.lower()))


def tokenize(text: str) -> list[str]:
    """
    Tách token trên văn bản ĐÃ BỎ DẤU: giữ chữ cái Latin và chữ số, bỏ hết còn lại.

    Không tự bỏ dấu ở đây — đầu vào phải là `text_folded`/`frame_text_folded`,
    để chỉ có MỘT cài đặt bỏ dấu trong toàn hệ (`strip_diacritics`).

    ⚠️ Truyền văn bản CÒN DẤU vào là hỏng ÂM THẦM, không ồn ào: regex loại mọi
    ký tự có dấu nên "đốt" → "t", "cháy" → "ch","y". Truy vấn biến thành rác,
    không có ngoại lệ nào được ném, chỉ có điểm thấp. Lỗi này ĐÃ XẢY RA THẬT
    trong `03_score_retrieval.py` và làm trung vị hạng của video đúng tụt xuống
    107.5/865. `Bm25Index.search` cảnh báo khi phát hiện — xem ở đó.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Bm25Hit:
    doc_id: str
    score: float


class Bm25Index:
    """
    Chỉ mục BM25 với danh sách ngược — chấm điểm chỉ chạm tài liệu CÓ CHỨA token
    truy vấn, không quét toàn bộ kho.

    Bất biến: `doc_id` duy nhất. Trùng `doc_id` ⟹ `ValueError` lúc dựng, vì bản
    ghi sau ghi đè bản trước là hỏng âm thầm (mất tài liệu, không có lỗi nào).
    """

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        if k1 < 0:
            raise ValueError(f"k1 không được âm, nhận {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b phải trong [0,1], nhận {b}")
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = []
        self._index: dict[str, str] = {}          # doc_id → vị trí (kiểm trùng)
        self._lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}   # token → [(doc, tf)]
        self._avgdl: float = 0.0
        self._idf: dict[str, float] = {}
        self._frozen = False

    @classmethod
    def build(cls, docs: Iterable[tuple[str, str]],
              k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> "Bm25Index":
        """Dựng từ `(doc_id, văn_bản_đã_bỏ_dấu)`."""
        idx = cls(k1=k1, b=b)
        for doc_id, text in docs:
            idx.add(doc_id, tokenize(text))
        idx.freeze()
        return idx

    def add(self, doc_id: str, tokens: Sequence[str]) -> None:
        if self._frozen:
            raise RuntimeError("chỉ mục đã freeze — không thêm tài liệu được nữa")
        if doc_id in self._index:
            raise ValueError(f"doc_id trùng lặp: {doc_id!r}")
        pos = len(self._doc_ids)
        self._index[doc_id] = doc_id
        self._doc_ids.append(doc_id)
        self._lengths.append(len(tokens))
        for tok, tf in Counter(tokens).items():
            self._postings.setdefault(tok, []).append((pos, tf))

    def freeze(self) -> None:
        """Chốt chỉ mục và tính sẵn IDF. Gọi lại lần nữa không sao."""
        if self._frozen:
            return
        n = len(self._doc_ids)
        self._avgdl = (sum(self._lengths) / n) if n else 0.0
        for tok, postings in self._postings.items():
            df = len(postings)
            self._idf[tok] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        self._frozen = True

    def __len__(self) -> int:
        return len(self._doc_ids)

    @property
    def avgdl(self) -> float:
        return self._avgdl

    def idf(self, token: str) -> float:
        return self._idf.get(token, 0.0)

    def search(self, query: str, top_k: int | None = None) -> list[Bm25Hit]:
        """
        Chấm điểm truy vấn (văn bản đã bỏ dấu) và trả về hit sắp GIẢM DẦN.

        Chỉ trả tài liệu có điểm > 0 — tài liệu không chứa token nào của truy vấn
        không phải "liên quan thấp", nó là "không có ý kiến", và đưa nó vào danh
        sách với điểm 0 chỉ làm loãng bước phân bổ slot phía sau.

        Phá hoà theo `doc_id` tăng dần ⟹ kết quả TẤT ĐỊNH, so được giữa hai lần
        chạy. Không có tính chất này thì mọi phép A/B đều nhiễu.
        """
        if not self._frozen:
            self.freeze()

        if looks_unfolded(query):
            # Không tự sửa: bỏ dấu ngầm ở đây sẽ tạo cài đặt bỏ dấu THỨ HAI, và
            # hai cài đặt sẽ lệch nhau đúng ở các ca khó. Chỉ hét lên.
            logger.warning(
                "Truy vấn CÒN DẤU tiếng Việt: %r — chỉ mục dựng trên văn bản đã "
                "bỏ dấu, nên token sẽ bị xé vụn và điểm gần như vô nghĩa. "
                "Gọi strip_diacritics() trước khi search().",
                query[:60],
            )

        scores: dict[int, float] = {}
        for tok in set(tokenize(query)):
            postings = self._postings.get(tok)
            if not postings:
                continue
            idf = self._idf[tok]
            for pos, tf in postings:
                dl = self._lengths[pos]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl) \
                    if self._avgdl else tf + self.k1
                scores[pos] = scores.get(pos, 0.0) + idf * (tf * (self.k1 + 1.0)) / denom

        hits = [Bm25Hit(self._doc_ids[p], s) for p, s in scores.items() if s > 0.0]
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return hits[:top_k] if top_k is not None else hits
