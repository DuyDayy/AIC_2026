# Quy trình ngày thi — 2 giờ 30, một lần chạy

> Đề của BTC chỉ có **đúng giờ thi**. Không chạy thử trước được, không sửa lại được.
> Trang này viết để **làm theo từng dòng**, không phải để đọc hiểu.

---

## 0. Làm TRƯỚC ngày thi

```bash
git pull && python -m pytest tests/ -q         # phải thấy 779 passed
modal run scripts/run.py --dir queries_smoke --out /tmp/smoke --spread 7
```

`queries_smoke/` là 3 file `.txt` bất kỳ. Lượt này **hâm nóng container Modal và nạp
model vào volume** — nếu để tới giờ thi mới chạy lần đầu thì mất thêm vài phút khởi động
nguội, và đó là vài phút không có gì bù lại.

Kiểm ba thứ, thiếu một là dừng và sửa trước:

| kiểm | phải thấy |
|---|---|
| `ls data/embed/` | có `.npy` chỉ mục và `query_cache.npz` |
| `python -c "import modal; print(modal.__version__)"` | chạy được, đã đăng nhập |
| dung lượng trống | ≥ 5 GB — chỉ mục 512 chiều chiếm 0,36 GB khi nạp |

---

## 1. Khi có đề — 3 phút đầu

Mỗi truy vấn là **một file `.txt`**, tên file thành `query_id`. Loại đề đoán theo:
tên file chứa `trake`/`qa`/`kis` → nội dung có `E1:`/`E2:` → còn lại là KIS.

```bash
mkdir -p queries_thi && rm -f queries_thi/*.txt
# … chép đề vào, mỗi câu một file .txt …
ls queries_thi/*.txt | wc -l          # ĐỐI CHIẾU với số đề BTC ra
```

⚠️ **Nếu đề có `E1:` `E2:` mà tên file không ghi `trake`** thì hệ tự nhận là TRAKE. Đúng.
Nhưng nếu đề TRAKE **không** dùng ký hiệu đó, phải **đổi tên file** thành `…-trake-….txt`,
không thì nó bị chấm như KIS và mất sạch câu đó.

---

## 2. Chạy — MỘT lệnh

```bash
modal run scripts/run.py --dir queries_thi --out nop_thi --spread 7
```

🔴 **`--spread 7` KHÔNG được quên.** Mặc định là `1`, và đo được nó **mất 3,3 lần điểm**
(0,4477 → 0,1366). Đây là lỗi tốn kém nhất có thể mắc trong cả ngày.

Chạy nền và ghi log ra file để đọc được tiến trình:

```bash
modal run scripts/run.py --dir queries_thi --out nop_thi --spread 7 > thi.log 2>&1 &
tail -f thi.log
```

---

## 3. Ngân sách thời gian — đo thật, 100 câu

| tầng | thời gian | ghi chú |
|---|---|---|
| khởi động Modal | ~30s | ~2–3 phút nếu container nguội |
| mã hoá đề trên GPU | ~40s | đề mới nên **không** có trong cache |
| nạp chỉ mục + 4 nguồn | **~40s** | cố định, không theo số câu |
| ②③ chấm 4 nguồn | **~135s** | phần tốn nhất; BM25 chiếm đa số |
| ⑤a dựng rổ | 4s | |
| ⑤c VLM chấm | ~60s | |
| ⑦ ghi bài nộp | vài giây | |
| **tổng** | **~5 phút** | |

**5 phút cho 100 câu.** Trong 2h30 chạy được **nhiều lượt**, nên còn dư giờ để kiểm và
chạy lại nếu cần.

⚠️ Phí cố định ~80 giây (nạp chỉ mục + nguồn) trả **mỗi lần gọi `modal run`**.
**Gom lô** — đừng chạy từng câu một.

### Thiếu giờ thì cắt theo thứ tự này

| bỏ | lệnh | mất gì |
|---|---|---|
| ⑤c VLM | `--vlm-top-k 0` | ~60s. Mất phần đẩy hạng của reranker |
| cả ⑤ | `--no-rerank` | thêm chút nữa |
| ba nguồn văn bản | `--light` | ~120s, nhưng **mất dung hợp** — đừng dùng trừ khi hết cách |

**Không bao giờ cắt `--spread`.** Nó không tốn giờ (chạy tại máy, vài giây) mà đổi 3,3 lần điểm.

---

## 4. Kiểm bài nộp TRƯỚC khi nộp

```bash
ls nop_thi/*.csv | wc -l                       # = số đề
awk -F, 'END{print NR}' nop_thi/*.csv          # mỗi file ≤ 100 dòng
python scripts/eval/score_submission.py --sub nop_thi --gt <gt nếu có>
```

Không có ground truth thì ít nhất kiểm **định dạng**:

```bash
python - <<'EOF'
import csv, sys
from pathlib import Path
bad = []
for p in sorted(Path("nop_thi").glob("*.csv")):
    rows = [r for r in csv.reader(open(p, encoding="utf-8")) if r and r[0]]
    if not rows:                       bad.append(f"{p.name}: RỖNG")
    if len(rows) > 100:                bad.append(f"{p.name}: {len(rows)} dòng > 100")
    if len(set(map(tuple, rows))) != len(rows): bad.append(f"{p.name}: có dòng TRÙNG")
    for r in rows[:5]:
        if not r[0].startswith("L") or not r[1].lstrip("-").isdigit():
            bad.append(f"{p.name}: dòng lạ {r}"); break
print("\n".join(bad) if bad else f"✓ {len(list(Path('nop_thi').glob('*.csv')))} file HỢP LỆ")
EOF
```

Cột thứ hai là **`frame_idx`** — số khung THẬT của video, không phải `n` (số thứ tự
keyframe). Đo được 0/173.426 khung có hai giá trị bằng nhau, lệch trung vị 5.267.

---

## 5. Hỏng thì làm gì

| triệu chứng | nguyên nhân đã gặp | xử lý |
|---|---|---|
| `IndexError` lúc khởi động | không có | — |
| treo ở khâu mã hoá mảnh cắt | bậc 1 bật | phải đang TẮT (`crop = 0`); nếu bật thì `--no-rerank` |
| `frame_id ≥ số khung` | rải vượt cuối video | đã chặn bằng `load_video_last_frame()`; nếu vẫn gặp thì `--spread 1` rồi nộp |
| máy hết RAM | nạp chỉ mục 1024 chiều | dùng `--dim 512` (mặc định) |
| Modal lỗi mạng giữa chừng | — | chạy lại; vector đề **đã cache** nên không tốn GPU lần hai |
| ra ít hơn số đề | file `.txt` rỗng hoặc sai encoding | kiểm `wc -l queries_thi/*.txt` |

**Chạy lại luôn an toàn.** Đầu ra ghi đè thư mục `--out`, và vector đề cache theo băm nội
dung nên lượt hai không tốn GPU.

---

## 6. Cấu hình đang chốt

| tham số | giá trị | vì sao |
|---|---|---|
| `dim` | 512 | 1024 không hơn (Δ=−0,002, KTC chứa 0) mà tốn gấp đôi RAM |
| trọng số 4 nguồn | `1 / 0,0891 / 0,1322 / 0,2128` | δ ASR ×2 sau kiểm chéo |
| `POOL_PER_SOURCE` | 40 | mỗi nguồn tự đề cử, rổ ~155 khung |
| `POOL_PER_VIDEO` | 10 | chốt an toàn chặn nguồn phẳng |
| `RERANK_WEIGHTS` | `fused4 1,0 · crop 0,0 · vlm 0,25` | bậc 1 đo được phá điểm |
| `VLM_TOP_K` | 40 | hạng `fused4` của đáp án trong rổ max = 40 |
| `--spread` | **7** ← phải gõ tay | mặc định 1 mất 3,3 lần điểm |

---

## 7. Điểm tham chiếu

Trên 100 truy vấn gán nhãn tay, chấm đúng luật BTC, `L = 11`:

| | Final |
|---|---|
| mốc = keyframe (lạc quan) | 0,6240 |
| mốc lệch ngẫu nhiên (bi quan) | 0,4886 |

Đề thi thật nằm đâu đó giữa hai số, **và nhiều khả năng thấp hơn cả hai** — bộ 100 câu do
chính đội viết nên câu chữ hợp với hệ hơn đề của người ngoài.
