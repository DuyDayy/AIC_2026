# Quy trình ngày thi — 2 giờ 30, một lần chạy

> Đề của BTC chỉ có **đúng giờ thi**. Không chạy thử trước được, không sửa lại được.
> Trang này viết để **làm theo từng dòng**, không phải để đọc hiểu.

---

## 0. Làm TRƯỚC ngày thi

```bash
git pull && python -m pytest tests/ -q         # bản clone: 380 passed · máy dev: 788
modal run scripts/run.py --dir queries_smoke --out /tmp/smoke
```

Hai con số vì kho chỉ mang **14 file test phụ thuộc mã đang vận hành** (380 test); 8 file
còn lại kiểm mã ngoài đường chạy nên nằm local. Bản clone mới thấy 380 là ĐÚNG, không phải
thiếu.

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
modal run scripts/run.py --dir queries_thi --out nop_thi
```

🟢 **Không có tham số `--spread`, và đó là thiết kế.** ⑦ nộp một khung mỗi mốc, 100 mốc.
Đừng đi tìm cờ để bật rải — không có, và ba test trong `tests/test_coverage.py` chặn việc
thêm lại. Lý lẽ ở `tests/test_coverage.py`; đánh đổi ở mật độ hiện tại là −75% trên GT v2.

Chạy nền và ghi log ra file để đọc được tiến trình:

```bash
modal run scripts/run.py --dir queries_thi --out nop_thi > thi.log 2>&1 &
tail -f thi.log
```

---

## 3. Ngân sách thời gian — đo thật

Đề dự kiến **30–40 câu**. Đo trên các lượt chạy thật rồi ngoại suy:

| tầng | 35 câu | 100 câu | co giãn theo số câu? |
|---|---|---|---|
| khởi động Modal | 30s | 30s | không |
| mã hoá đề trên GPU | 14s | 40s | có |
| **nạp chỉ mục + 4 nguồn** | **80s** | **80s** | **KHÔNG** |
| ②③ chấm 4 nguồn | 47s | 135s | có |
| ⑤a dựng rổ | 2s | 4s | có |
| ⑤c VLM (cả rổ) | **~250s** | ~790s | có |
| ⑦ ghi bài nộp | 2s | 5s | có |
| **tổng** | **~7 phút** | **~19 phút** | |

Lượt chạy 100 câu gần nhất (`submission_now`, không rải + VLM 160) đo **848 giây = 14 phút**
đồng hồ tường, trong đó `②③` 134s và `⑤a` 4s là số in ra, phần còn lại ~555s là `⑤c`. Tức
`⑤c` một mình chiếm **65%** một lượt chạy, và nó là tầng duy nhất đáng bàn về giờ.

Từ nay **không phải đọc log để biết giờ**: `_report.json` có trường `timing_s` với mốc từng
tầng, gồm `⑤c VLM bậc 2` và `⑦ phát bài nộp` (hai mốc trước đây thiếu — nên tầng ĐẮT NHẤT
là tầng duy nhất không đo được).

🟢 **Đường ống dùng ~5% ngân sách 2h30.** Chạy được ~20 lượt. Không có lý do gì phải cắt
tầng nào vì thiếu giờ — bảng cắt bên dưới chỉ dùng khi Modal trục trặc.

⚠️ **⑤b mảnh cắt tốn 655 giây** (201s cắt + 454s mã hoá) nếu ai bật lại bằng `--crop-w`.
Đó là 11 phút để đổi lấy `+0,0009` với KTC chứa 0 — **đừng bật trong ngày thi**.

🔴 **MẤT MẠNG GIỮA BÀI CHẠY — ĐÃ GẶP THẬT HAI LẦN.** Biểu hiện là
`ConnectionError: [Errno 8] nodename nor servname provided` rồi
`App state is APP_STATE_STOPPED`. Bài chạy mất sạch, không có bài nộp một phần.

- **Chạy sớm, đừng để sát giờ nộp.** Một lần mất mạng là mất trọn lượt chạy.
- Kiểm bằng `modal app list` — có kết quả là mạng đã về.
- Chạy lại **an toàn và rẻ**: vector đề cache theo băm nội dung nên lượt hai không tốn GPU.
- Đừng đi sửa mã vì `APP_STATE_STOPPED` — nó là **hậu quả**, không phải nguyên nhân.

🔴 **THỜI GIAN BẬC 2 KHÔNG ỔN ĐỊNH.** Cùng mã, cùng dữ liệu, hai lần chạy 110 câu ×
160 khung đo được **13 phút** và **>40 phút**. Nguyên nhân ở phía Modal (số container cấp
được, tình trạng nguội/ấm), không ở mã ta. Hệ quả cho ngày thi:

- **Chạy bậc 2 SỚM**, đừng để sát giờ nộp.
- Nếu sau **10 phút** mà tiến trình `⑤c` chưa quá nửa: còn giờ thì cứ chờ; sắp hết giờ thì
  **dừng, chạy lại với `--vlm-top-k 0`** và nộp bản không có bậc 2. Mất ~13% điểm tương
  đối, nhưng **không nộp được thì mất 100%**.
- Luôn giữ một bài nộp **đã hoàn chỉnh** trước khi thử cấu hình đắt hơn.

⚠️ **Phí cố định 80 giây chiếm 41% một lượt 35 câu** (so với 23% ở 100 câu). Gom cả đề
vào **một lần gọi**; chạy từng câu một là trả 80 giây × số câu.

🟢 **Dư giờ nên dùng để KIỂM, không phải để chạy nhiều cấu hình.** Không có ground truth
lúc thi nên chạy 5 cấu hình rồi cũng không biết chọn cái nào — chỉ tổ rối. Chạy **một**
cấu hình đã chốt, rồi dành giờ soi bài nộp bằng mắt.

### Thiếu giờ thì cắt theo thứ tự này

| bỏ | lệnh | mất gì |
|---|---|---|
| ⑤c VLM | `--vlm-top-k 0` | tiết kiệm ~4 phút nhưng **mất 13% điểm tương đối** — chỉ cắt khi Modal hỏng |
| cả ⑤ | `--no-rerank` | thêm chút nữa |
| ba nguồn văn bản | `--light` | ~120s, nhưng **mất dung hợp** — đừng dùng trừ khi hết cách |

Không còn tham số rải nào để cắt.

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
| `frame_id ≥ số khung` | không còn xảy ra — chỉ nộp keyframe thật | — |
| máy hết RAM | nạp chỉ mục 1024 chiều | dùng `--dim 512` (mặc định) |
| `ConnectionError: nodename nor servname` | **mất DNS/mạng** — đã gặp thật 2 lần | kiểm `modal app list`; có kết quả là mạng đã về, chạy lại. Vector đề **đã cache** nên không tốn GPU lần hai |
| `App state is APP_STATE_STOPPED` | **hậu quả** của mất mạng, không phải lỗi mã | như trên. Đừng đi sửa mã vì lỗi này |
| Modal lỗi mạng giữa chừng | — | chạy lại; vector đề **đã cache** nên không tốn GPU lần hai |
| ra ít hơn số đề | file `.txt` rỗng hoặc sai encoding | kiểm `wc -l queries_thi/*.txt` |

**Chạy lại luôn an toàn.** Đầu ra ghi đè thư mục `--out`, và vector đề cache theo băm nội
dung nên lượt hai không tốn GPU.

---

## 5b. Định dạng bài nộp — thể lệ chỉ cho 3 lượt mỗi gói

`scripts/run.py` tự ghi CSV theo thể lệ và tự đóng gói: `submission/*.csv` →
`submission.zip` với tiền tố `submission/` bên trong. **Không tự nén tay** — `zip -r`
gói cả `_report.json` và `_rerank_scores.npz` (hàng chục MB), còn nén trực tiếp các CSV
là "lỗi thường gặp số 2" của BTC và chỉ lộ ra sau khi đã tiêu một lượt.

Kiểm nhanh trước khi tải lên:

```bash
unzip -l submission.zip                 # MỌI dòng phải bắt đầu bằng submission/
python - <<'EOF'
import csv, io, zipfile
for n, in [(n,) for n in zipfile.ZipFile("submission.zip").namelist()]:
    t = zipfile.ZipFile("submission.zip").read(n).decode("utf-8")
    rows = [r for r in csv.reader(io.StringIO(t)) if r]
    want = {"kis": 2, "qa": 3}.get(n.rsplit("-", 1)[-1][:-4])
    print(f"{n:<34} {len(rows):>3} dòng · số cột {sorted({len(r) for r in rows})}"
          + ("  ✗ SAI" if want and {len(r) for r in rows} != {want} else ""))
EOF
```

`_report.json` có khối `format`: `ghi_chu` liệt kê mọi đáp án bị cắt/trim,
`khong_ghi_duoc` liệt kê truy vấn không ra được file. **Cả hai phải rỗng.**

---

## 6. Cấu hình đang chốt

| tham số | giá trị | vì sao |
|---|---|---|
| `dim` | 512 | 1024 không hơn (Δ=−0,002, KTC chứa 0) mà tốn gấp đôi RAM |
| `alpha` (③ modality) | `visual 1/3 · ocr 1/3 · asr 1/3` | ĐỀU. Nguồn `object` đã gỡ; chuẩn hoá z đã bỏ, ③ hợp bằng RRF |
| `beta` (③ expansion) | đều trong từng modality — 3 run ⟹ `1/3` | mỗi bản mở rộng đóng góp ngang bản gốc |
| `RRF_K` | 60 | **chưa quét lại** trong chế độ không cắt danh sách |
| rổ ⑤a | **top-`POOL_CAP` của điểm ③ đã hợp** | bỏ đề cử riêng mỗi run: độ phủ 89,0% → **98,0%** = trần lý thuyết. Δ+0,0900, KTC95 [+0,04, +0,15], thắng 9/thua 0 |
| `POOL_CAP` | 300 | nay là phép cắt CHỦ ĐỘNG, không còn là chốt an toàn — **chưa quét** |
| `POOL_PER_VIDEO` | **rời đường chạy** | mất mát thuần: pv=10 kéo độ phủ 98,0% → 92,0% |
| `TRAKE_K_PER_VIDEO` | 1 | ⚠️ chỉ điền **26/100** dòng. `k=10` điền đủ 100 ở 92% câu — chưa đo được ĐIỂM (bộ TRAKE có 6 câu) |
| `RERANK_WEIGHTS` | `fused4 1,0 · crop 0,0 · vlm 0,25` | crop trống cả ở nhóm ≥2 màu, tốn 655s |
| `VLM_TOP_K` | **160** (cả rổ) | bỏ VLM mất 13% điểm tương đối; K=30→160 mua thêm 0,017 |
| rải khung | **không có, theo thiết kế** | ⑦ nộp 1 khung/mốc; KHÔNG có cờ, test chặn thêm lại. Ở mật độ hiện tại việc này **mất 0,3381 Final** (0,1115 so với 0,4496) — chỉ hoà lại khi cắt dày hơn |

---

## 7. Điểm tham chiếu

Trên 100 truy vấn gán nhãn tay, chấm đúng luật BTC, `L = 11`:

| | Final |
|---|---|
| mốc = keyframe (lạc quan) | 0,5540 |
| mốc lệch ngẫu nhiên (bi quan) | 0,4701 |

Trên bộ **110 câu giữ kín** (chưa dùng để chỉnh gì, câu ngắn hơn nên khó hơn):
`L=11` bi quan = **0,2778**. Hai bộ **không so ngang được**.

Đề thi thật nằm đâu đó giữa hai số, **và nhiều khả năng thấp hơn cả hai** — bộ 100 câu do
chính đội viết nên câu chữ hợp với hệ hơn đề của người ngoài.

### Bộ 100 câu GT v2, cấu hình HIỆN TẠI (không rải), `L = 11`

| thước | Final |
|---|---|
| cấp khung, mốc lệch | **0,1115** |
| cấp khung, mốc lệch, bản CÓ rải | 0,4496 |
| cấp shot (`score_shots.py`) | **0,7120** |

🔴 Ba con số này phải đọc cùng nhau. Cấp shot `R@1 = 0,56` nghĩa là hệ **xếp đúng shot ở
hạng 1 cho 56% câu**; cấp khung `R@1 = 0,05` nghĩa là nó gần như luôn trượt cửa sổ 11 khung.
Chỗ hổng là **độ phân giải thời gian**, không phải xếp hạng — 50/100 câu mất điểm dù hạng 1
đã đúng shot.
