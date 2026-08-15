"""
Bỏ dấu tiếng Việt — MỘT cài đặt duy nhất cho toàn hệ
======================================================

Tách khỏi `transcript_store` vì nó không liên quan gì tới transcript: nó là phép chuẩn
hoá chuỗi mà **cả bốn nguồn điểm** và mọi chỉ mục BM25 đều đi qua.

=============================================================================
VÌ SAO PHẢI LÀ MỘT, KHÔNG ĐƯỢC HAI
=============================================================================

Chỉ mục và truy vấn phải bỏ dấu theo **cùng một quy ước**, nếu không token không bao giờ
khớp. Hai cài đặt lệch nhau ở một ký tự (chữ `đ`, chữ hoa, ký tự tổ hợp) là đủ để điểm
tụt mà **không có gì báo** — mọi thứ vẫn chạy, chỉ là không khớp.

Lỗi này đã xảy ra thật: đưa truy vấn CÒN DẤU vào chỉ mục đã bỏ dấu làm nguồn `object`
chấm ra trung vị hạng **14.407** thay vì 3.653. `bm25.looks_unfolded` bắt được.

Nên `bm25.py` cố tình **không tự bỏ dấu**: nó ép chỗ gọi phải dùng hàm này.

=============================================================================
QUY ƯỚC
=============================================================================

Khớp `text_ascii_folded` của OCR và `*_folded` của `objects-full`, tức chuẩn NFD → bỏ
dấu tổ hợp → `đ`/`Đ` thành `d`/`D` → chữ thường. `đ` phải xử lý riêng vì nó KHÔNG phải
`d` + dấu tổ hợp trong Unicode, nên NFD không tách nó ra.
"""

from __future__ import annotations

import unicodedata


def strip_diacritics(s: str) -> str:
    """Bỏ dấu tiếng Việt, trả chữ thường. Xem docstring module về quy ước."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D").lower()
