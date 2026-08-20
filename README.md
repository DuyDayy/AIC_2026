# Tầng truy vấn — AIC 2026

Hệ truy xuất video đa phương thức cho vòng sơ tuyển. Tiền xử lý (khung hình · ASR · OCR ·
vật thể) đã xong; kho này là **tầng truy vấn**: từ đề bài ra bài nộp.

Vận hành ngày thi: [`RUNBOOK.md`](RUNBOOK.md) · Kế hoạch tối ưu: [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md)

```
⓪ mã hoá offline   173.426 khung → lưu 1024 chiều, tra ở 512
       │
đề → ① PROBE + mở rộng ──→ ② 7 run chấm điểm ──→ ③ DUNG HỢP  RRF hai tầng
     tách mốc E1/E2          1 thị giác              beta: expansion trong modality
     đọc expansions/         3 OCR (gốc+2)           alpha: modality với nhau
                             3 ASR (gốc+2)                    │
                                    │                         │
                             ⑤a RỔ ─┘                         │
                             top RIÊNG mỗi run ───────────────┤
                                                              ▼
                                    ⑦ nộp ←── ④ DANTE ←── ⑤c RERANK
                                    1 khung/mốc    k-best     VLM, RRF tầng ba
```

**Rổ chọn ai vào vòng trong; RRF hai tầng quyết thứ hạng; DANTE ràng buộc thứ tự thời gian.**

Toàn bộ đường chạy xếp theo **HẠNG**, không có chuẩn hoá z ở bất kỳ tầng nào.

---

## Công nghệ

| tầng | công nghệ | vai trò |
|---|---|---|
| ⓪ | **jina-clip-v2** (0,9B, 89 ngôn ngữ) | nhúng chung ảnh–chữ; Matryoshka lưu 1024, tra 512 |
| ① | tách mốc `E1:`/`E2:`, rút trích dẫn | một truy vấn → N probe. Q&A: câu hỏi → câu **mô tả** |
| ① | mở rộng truy vấn | đọc `expansions/` — 2 bản OCR + 2 bản ASR mỗi đề, **không gọi LLM lúc thi** |
| ② | jina-clip · **BM25** ×2 | thị giác · OCR · lời nói. Mỗi modality văn bản chấm 3 lần: gốc + 2 mở rộng |
| ③ | **`hierarchical_rrf`** | RRF hai tầng — `beta` gộp expansion TRONG modality, `alpha` gộp modality |
| ⑤a | **rổ ứng viên** (`pool.py`) | mỗi RUN đề cử top 40 riêng → rổ 266–269 khung, **bộ lọc cứng** |
| ⑤b | mảnh cắt vật thể + jina-clip | **BỎ** — trống cả ở nhóm ≥2 màu, tốn 655s + $0,77 |
| ⑤c | **Qwen2.5-VL-7B** | `P(khung khớp)` = softmax trên hai token `1`/`0`; hợp vào nền bằng **RRF tầng ba** |
| ④ | **DANTE** k-best (`kbest.py`) | DP thứ tự thời gian `O(N·T)`; **KIS = TRAKE với N=1** |
| ⑥ | Qwen2.5-VL | chỉ đề Q&A: đọc khung + OCR + lời nói → sinh `answer` |
| ⑦ | một khung mỗi mốc, 100 mốc | rải đã XOÁ — chờ keyframe dày hơn; xoá rải mất 0,3129 ở mật độ hiện tại |

Hạ tầng: **Modal** — GPU A10G, `.map()` song song (mã hoá mảnh cắt nhanh **31×** so với
`.remote()` tuần tự: nghẽn là ĐƯỜNG TRUYỀN, không phải GPU).

---

## Từng tầng

### ⓪ Mã hoá offline — jina-clip-v2

| | |
|---|---|
| mô hình | `jinaai/jina-clip-v2` — 0,9B (561M tháp chữ + 304M tháp ảnh) |
| không gian | **chung cho ảnh và chữ**, 89 ngôn ngữ gồm tiếng Việt |
| chiều | lưu **1024**, tra **512** (Matryoshka, cắt lúc đọc) |
| quy mô | 173.426 khung · ~$4,32 mã hoá một lần |
| provenance | `artifacts/embed/embed/manifest.json` — SHA của model **và** của remote-code |

Một mô hình cho cả hai phương thức nên không cần dịch truy vấn. Lưu đủ 1024 rồi cắt **lúc
đọc** vì giá trị của Matryoshka là *quyền chọn số chiều sau*. Cắt **rồi mới** chuẩn hoá —
ngược lại thì vector mất tính đơn vị. Đo đầu-cuối: 1024 chỉ hơn `−0,002` (KTC chứa 0) mà
tốn gấp đôi RAM.

### ① Probe hoá

| | |
|---|---|
| đầu vào | một file `.txt` mỗi truy vấn |
| đầu ra | `N` probe — KIS/QA: `N=1` · TRAKE: `N` mốc |
| tách mốc | ký hiệu `E1:` `E2:` … |
| trích dẫn | chuỗi trong ngoặc kép, gắn vào **mốc chứa nó** |
| Q&A | `declarativize()` — câu hỏi → câu **mô tả** trước khi mã hoá |

Tháp chữ huấn luyện trên **caption**, mà câu nghi vấn hỏi về thứ *ta chưa biết*: trong
*"…cầm ly màu gì?"*, cụm "màu gì" là chỗ trống, không mô tả khung nào. ⑥ vẫn nhận câu hỏi
gốc vì nó cần biết đang được hỏi gì. *(Ý lấy từ NII-UIT @ VBS2025 mục 2.7.)*

### ② Ba nguồn điểm, BẢY lượt chấm

| nguồn | công nghệ | dữ liệu | phủ | số lượt |
|---|---|---|---|---|
| thị giác | jina-clip-v2, cosine | vector 512 chiều | 100% | 1 (probe) |
| OCR | **BM25** (`k1=0,6 · b=0,9`) | chữ hiện trên màn | 98% | 3 (gốc + `qo` ×2) |
| lời nói | **BM25** (`k1=1,5 · b=0,0`) | ASR trong cửa sổ **±5 giây** quanh khung | 97% | 3 (gốc + `qa` ×2) |

Mỗi **lượt** là một danh sách hạng độc lập đi vào ③ — tầng `beta` cộng ba lượt trong cùng
một nguồn, tầng `alpha` cộng ba nguồn. Nguồn **vật thể đã gỡ** (xem mục Trọng số).

BM25 **cố tình không tự bỏ dấu** — bỏ dấu sinh đụng độ thật: `đồng`(Nai)/`động`(vật),
`cán`/`căn`/`cân`. Mỗi nguồn trả `SourceScores(scores, covered)`, trong đó `covered` nói
nguồn **có dữ liệu** cho khung đó, **không** nói truy vấn khớp.

**`k1`/`b` RIÊNG từng nguồn** — ba nguồn muốn `b` ngược nhau. `b` là mức chuẩn hoá theo
**độ dài văn bản**; MRR của khung đáp án tại `k1=1,5`:

| `b` | OCR | ASR |
|---|---|---|
| 0,0 | 0,0194 | **0,1816** |
| 0,75 *(mặc định cũ)* | 0,1660 | 0,1505 |
| 0,9 | **0,1641** | 0,1520 |

Cơ chế: **OCR** là chữ chạy trên màn, độ dài rất chênh — khung nhiều chữ dễ khớp bừa nên
cần chuẩn hoá mạnh. **ASR** lấy trong cửa sổ ±5 giây **cố định** nên độ dài gần đều;
chuẩn hoá theo độ dài chỉ thêm nhiễu.

⚠️ Đầu-cuối chỉ **+0,005 với KTC chứa 0** (16/110 câu đổi). Giữ vì lý do **cơ chế** —
`b=0` cho ASR suy ra từ thiết kế nguồn, đúng bất kể bộ eval — chứ không vì con số.

⚠️ Nguồn **vật thể có MRR 0,0003**, hạng đáp án cỡ 3.000 — đó là lý do nó bị gỡ, chứ
không phải vì chỉnh sai tham số.

### ③ Dung hợp — RRF hai tầng

    Score(d) = Σ_m alpha_m · ( Σ_j beta_mj / (k + rank_mj(d)) )
    ràng buộc: Σ alpha_m = 1  ·  Σ_j beta_mj = 1 cho TỪNG modality

| | |
|---|---|
| hàm | `score_matrix.hierarchical_rrf` |
| `k` | 60 |
| `alpha` | **đều** — `visual 1/3 · ocr 1/3 · asr 1/3` |
| `beta` | **đều** trong từng modality — 3 run ⟹ `1/3` mỗi nhánh |
| số run | 7 = 1 thị giác + 3 OCR + 3 ASR |

Trọng số áp vào **đóng góp RRF**, không áp vào raw score — cosine và BM25 không bao giờ
được cộng trực tiếp với nhau.

**Vì sao hai tầng, không phẳng.** Ném cả 7 run vào một phép RRF phẳng thì modality nào
nhiều expansion hơn tự có nhiều quyền vote hơn: Visual một run bị OCR ba run lấn át chỉ
vì đếm run. Chuẩn hoá `beta` trong từng modality chặn đúng điều đó. [ĐO] trên bench_kis:

| cấu hình | MRR | R@100 |
|---|---|---|
| không expansion | 0,4751 | 0,91 |
| bỏ bản gốc, chỉ QE | 0,4867 | 0,93 |
| QE chỉ OCR | 0,4989 | 0,92 |
| QE chỉ ASR | 0,4874 | 0,92 |
| **phân cấp, QE cả hai** ★ | **0,5281** | **0,93** |
| **PHẲNG, cùng 5 run** | 0,4427 | 0,89 |

Phân cấp hơn phẳng **0,085 MRR** — khoản lớn nhất bảng. Và giữ bản gốc bên cạnh
expansion hơn bỏ nó 0,041: expansion **bổ sung**, không thay thế.

**Vì sao cả hai trọng số đều ĐỀU.** Không phải vì chưa thử. E4 quét trọng số toàn cục hai
tầng trên 210 câu tuning, báo trên 100 câu held-out chưa đụng tới:

```
alpha* tìm được = (0,30 · 0,35 · 0,35)  ≈ đều
held-out: đều 0,5281  →  có trọng số 0,4884
ΔMRR bắt cặp −0,0398 · KTC95 [−0,0783, −0,0056] · thắng 10 / thua 18
```

Khoảng tin cậy nằm **trọn dưới 0**: trọng số toàn cục **kém hơn có ý nghĩa**. Lệch khỏi
đều phải đo trên tập giữ kín trước, không phải đặt theo trực giác.

🔴 **Đánh đổi đã biết khi chọn RRF.** A/B với kiến trúc z-norm cũ trên 210 câu:
`ΔMRR −0,0375` KTC95 `[−0,0707, −0,0057]` (RRF **kém hơn** về MRR, có ý nghĩa) nhưng
`ΔR@100 +0,0095` và RRF thắng rõ ở `R@20`–`R@50` (holdout 0,409 → 0,509). RRF chỉ nhìn
**hạng** nên kéo được ứng viên từ sâu lên, nhưng vứt mất **biên độ** — mà biên độ chính
là thứ tách hạng 1 khỏi hạng 3. Chọn RRF là chấp nhận đổi `R@1` lấy `R@20`–`R@100`.

⚠️ **`k = 60` chưa có căn cứ trong chế độ đang chạy.** RRF nguyên bản định nghĩa trên
danh sách **đã cắt top-K**; `rrf_normalize` xếp hạng cả tập được phủ (173.426). Hệ quả:
tỉ lệ đóng góp hạng-1 so với hạng-cuối là **2844×** thay vì **2,6×** như khi cắt top-100.
[ĐO] không cắt vẫn tốt hơn cắt (MRR 0,2862 so với 0,2782 ở top-100, đơn điệu, bão hoà
quanh top-5000) — nên **không** cắt. Nhưng `k` kế thừa từ E3 và chưa từng được quét trong
chế độ này.

### ⑤a Rổ ứng viên — `src/retrieval/pool.py`

| | |
|---|---|
| cách chọn | mỗi RUN đề cử **top 40 của riêng nó**, hợp lại |
| kích thước | **266–269** khung = 7 run × 40 sau khử trùng (`POOL_CAP = 300`) |
| vai trò | **bộ lọc CỨNG** — chỉ khung trong rổ mới được nộp |
| chốt 1 | `score > 0` — khung có chữ nhưng không khớp từ nào thì không được đề cử |
| chốt 2 | **10 khung/video/nguồn** — chặn nguồn cho điểm phẳng cả video |
| lai lịch | `provenance` ghi nguồn nào đề cử khung nào |

Thị giác mang hệ số `1,0` còn ba nguồn kia `0,09–0,21`, nên khung chỉ có bằng chứng **thuần
OCR** gần như không nổi lên trong điểm đã hợp: đo được **10/100 câu** có video đúng vắng mặt
hoàn toàn, mà 9/10 nằm ở hạng 16–62 — với tới được, chỉ là không được với tới.

Chốt 2 sinh ra từ một ca thật: OCR khớp **logo kênh** hiện suốt video ⟹ điểm phẳng ⟹ top-40
của nó là 40 khung tuỳ tiện cùng một video.

**Bốn nguồn đề cử gần như RỜI NHAU** — đo trên 100 câu GT v2, đếm số nguồn cùng đề cử một
khung:

| số nguồn đồng thuận | khung | tỉ lệ |
|---|---|---|
| 1 | 15.516 | **98,6%** |
| 2 | 200 | 1,3% |
| 3 | 28 | 0,2% |
| 4 | 0 | 0% |

Hai hệ quả, cả hai đều là **lý lẽ cơ chế cho kết quả đã đo trước đó**:

- cho mỗi nguồn **suất riêng** có lợi `+0,0153` — vì không có suất thì ba nguồn yếu **vô
  hình**, chứ không phải "được ưu tiên thấp";
- dùng `agree` **làm điểm** thì hại `0,0208` — vì nó gần như hằng số 1, nên chỉ mang nhiễu.

🔴 **`POOL_CAP` đã từng âm thầm cắt mất 25% rổ.** Khi ③ còn 4 nguồn, rổ tối đa là
`4×40 = 160` nên `POOL_CAP = 200` không bao giờ chạm — README cũ ghi đúng như vậy. ③
chuyển sang RRF phân cấp thì mỗi truy vấn có **7 run**, rổ thô lên **266–269** [ĐO trên
gtv2], và 200 bắt đầu cắt ở **mọi** truy vấn. Phép cắt ấy xếp theo điểm **nền**, nên nó
xoá đúng phần mà suất riêng mỗi run sinh ra để bảo vệ — khoản `+0,0153`. Nâng lên **300**
(> 7×40 = 280) để chốt trở lại đúng vai trò chốt an toàn.

⚠️ Thêm expansion thứ ba mỗi modality (9 run = 360) thì phải nâng tiếp, nếu không lỗi này
quay lại y nguyên **và vẫn im lặng**.

### ⑤b Bậc 1 — mảnh cắt vật thể ⚠️ TẮT

| | |
|---|---|
| công nghệ | cắt bbox vật thể → mã hoá bằng jina-clip → cosine với truy vấn |
| sinh ra để chữa | *attribute binding* — đảo màu giữa hai vật chỉ đổi vector chữ **0,026** |
| lợi trên đề màu | 54% → **77%** đúng màu (8 câu thuần màu, chấm bằng mắt) |
| **trên bộ 100 câu** | **−0,030** · 88/100 lớp kiểm chéo chọn `crop = 0` |
| trạng thái | **TẮT** (`RERANK_WEIGHTS["crop"] = 0`) — trọng số 0 ⟹ không tính luôn |

Tháp chữ mã hoá `{áo, xe, đỏ, trắng}` như **túi khái niệm**; cắt vật ra mã hoá riêng buộc
màu vào vật bằng cấu trúc.

🔴 **Lời giải thích cũ của tôi ở đây SAI và đã bị thay.** Tôi từng viết hai phép đo không
mâu thuẫn vì *"bộ 100 câu gần như không có câu phụ thuộc màu"*. Đo lại thì màu **rất phổ
biến**:

| bộ | có tên màu | **≥2 màu** (ca binding thật) |
|---|---|---|
| GT v2 | **58%** | 15% |
| 100 câu tay | 38% | 14% |
| 110 giữ kín | 43% | 16% |

Nên ⑤b hại **dù màu phổ biến** — không bào chữa được bằng thành phần bộ eval. Cách đọc
đúng: phép đo 54%→77% chạy trên **8 câu thuần màu, chấm bằng mắt trên 6 khung đầu**, còn
đường ống đo *"khung đáp án có được xếp hạng đầu không"*. ⑤b có thể làm màu đúng hơn
**mà vẫn** làm hạng của khung đáp án tệ đi.

**Phép đo chốt hạ: bật crop, tách theo số màu trong câu** (GT v2, `L=11`, bắt cặp,
bootstrap 4000):

| nhóm | n | crop TẮT | crop BẬT | Δ | KTC95 | T/Th |
|---|---|---|---|---|---|---|
| ≥2 màu (binding thật) | 15 | 0,0717 | 0,0733 | +0,0017 | [−0,0037, +0,0087] | 1/1 |
| đúng 1 màu | 43 | 0,1298 | 0,1301 | +0,0003 | [−0,0054, +0,0080] | 2/5 |
| không màu | 42 | 0,1243 | 0,1254 | +0,0012 | [−0,0045, +0,0071] | 6/5 |
| **tất cả** | 100 | 0,1187 | 0,1196 | +0,0009 | [−0,0027, +0,0051] | 9/11 |

⑤b **chết ở đúng nhóm nó sinh ra để chữa**: nhóm ≥2 màu được +0,0017, KTC chứa 0, thắng 1
thua 1 trên 15 câu. Đây không còn là "chưa đo đúng phạm vi" — phạm vi đã đo đúng và vẫn
trống. Chi phí đối lại: **655 giây** (201s cắt + 454s mã hoá, tức ~11 phút của ngân sách
2h30) và **$0,77**. Kết luận: **bỏ**, không phải "tắt tạm".

Bật lại bằng `crop > 0` nếu muốn đo lại; cổng chặn ở `run.py` đảm bảo trọng số 0 thì
không mã hoá mảnh nào, nên tắt là **thật sự** không tốn gì.

### ⑤c Bậc 2 — Qwen2.5-VL

| | |
|---|---|
| mô hình | `Qwen/Qwen2.5-VL-7B-Instruct`, bfloat16, GPU A10G |
| cách chấm | ép trả **một ký tự** `1`/`0`, đọc **softmax trên đúng hai token đó** |
| phạm vi | `VLM_TOP_K = 160` — **cả rổ** |
| trọng số | `0,25` so với `1,0` của dung hợp |
| ảnh gửi | 640px, JPEG q85 |

Không sinh chuỗi, không phân tích văn bản ⟹ điểm **liên tục và tất định**. Bậc này thấy thứ
mảnh cắt không thấy: quan hệ giữa các vật, hành động, phủ định.

⚠️ Đây là **cú can thiệp mạnh và hiếm**, không phải bộ chỉnh êm: 85/110 câu nó không đụng
tới, lợi ích đến từ **18 thắng / 7 thua**. Tỉ lệ 2,6 ăn 1 là lý do trọng số giữ ở `0,25` —
nâng lên là khuếch đại cả phần đúng lẫn phần sai.

### ④ DANTE k-best — `src/submission/kbest.py`

| | |
|---|---|
| hệ thức | `DP[i,t] = S[i,t] + max_{τ<t}( DP[i−1,τ] − λ·(t−τ) )` |
| độ phức tạp | `O(N·T)` nhờ **max luỹ tiến**, thay vì `O(N·T²)` |
| trục thời gian | **mili-giây** — `n` bước không đều, `frame_idx` khác fps giữa video |
| λ | **0** — mọi `λ > 0` đo được đều tệ hơn |
| k-best | `k` đường mỗi video rồi **sắp toàn cục** |

`N = 1` làm hệ thức suy biến thành `max_t S[0,t]`, tức **KIS chính là TRAKE với N=1**. ⑦ chỉ
có **một** đường mã, khoá bằng `tests/test_kis_is_trake_n1.py`.

⚠️ Nhánh TRAKE cũ gọi `dante_over_videos` vốn trả *một đường mỗi video* — tức khử trùng theo
video ngầm, đo được **−12,0pp**.

### ⑥ Đầu đọc QA

| | |
|---|---|
| mô hình | Qwen2.5-VL-7B, ảnh 896px |
| đầu vào | khung top-1 **+ OCR của khung + lời nói quanh nó (±5 giây)** |
| đầu ra | chuỗi `answer` ghép vào cột thứ ba của bài nộp |

Nhiều câu hỏi (tên riêng, con số, ngày tháng) **chỉ đọc được từ chữ**, không nhìn ra từ ảnh.
Thiếu tầng này thì câu Q&A được **0 điểm** dù tìm đúng khung — thể lệ đòi `aᵢ = GTₐ`.

⚠️ Tầng này **chưa có phép đo nào**.

### ⑦ Nộp bài — MỘT khung mỗi mốc, KHÔNG rải

| | |
|---|---|
| ngân sách | 100 dòng/truy vấn = **100 mốc** |
| mỗi mốc | đúng **một** dòng: `frame_idx` thật của keyframe |
| khử trùng `(video, frame)` | **có** — `R@k = max`, hai dòng giống hệt mua đúng một cơ hội |
| khử trùng theo cảnh/video | **KHÔNG** — đo được −4,3pp / −12,0pp |

**Một khung mỗi mốc là THIẾT KẾ, không phải tuỳ chọn — và không có cờ để bật lại rải.**

Vì sao không để cờ: rải không phải cách xếp hạng tốt hơn, nó là cách **tiêu 100 suất ngân
sách để đắp khoảng trống giữa các keyframe**. Khoảng trống đó là thuộc tính của **bộ
keyframe**, không phải của tầng truy vấn — nên chỗ giải nó là lúc **cắt khung**. Một cờ ở
đây chỉ mời người ta lấy ngân sách đắp cho dữ liệu thiếu, rồi tưởng mình đang chỉnh mô
hình. Ba test trong `tests/test_coverage.py` khoá quyết định này: `run.py` không được
import `coverage`, `main()` không được có tham số nào dính `spread`/`rai`, và
`coverage.py` **không được xoá**.

Đánh đổi thì nói rõ, không giấu — ở mật độ keyframe hiện tại thiết kế này **kém hơn**:

| bộ đề | một khung mỗi mốc | rải | Δ |
|---|---|---|---|
| GT v2 100 câu, `L=11` | **0,1115** | 0,4496 | −75% |
| 110 giữ kín | 0,0857 | 0,2334 | −63% |

Nguyên nhân là **hình học**: keyframe cách nhau trung vị **48 khung** còn cửa sổ đáp án
thể lệ ~10 khung, nên xác suất một cửa sổ chứa sẵn keyframe của ta chỉ **23,5%**.

Trần đó là **hàm của mật độ**, và kế hoạch cắt dày hơn hoá giải đúng nó:

| khe keyframe | trần `L=9` | trần `L=11` |
|---|---|---|
| 48 (hiện tại) | 23,5% | 28,6% |
| 24 (×2 dày) | 45,7% | 53,1% |
| **12 (×4 dày)** | **73,5%** | **81,1%** |
| 10 (mỗi khung thứ 10) | 90,0% | 100% |

Ở ×4 dày, mỗi keyframe tự phủ cửa sổ của nó và rải thành vô nghĩa. Hai thứ **không cộng
dồn, chúng thay nhau** — nên thiết kế này đặt cược vào mật độ, và cược đó chỉ thắng sau khi
bộ keyframe dày hơn có thật.

Cùng hướng đó, khối dựng `win[row]` (nửa khe ∩ biên cảnh ∩ cuối video) **đã rời khỏi
`run.py`**: nó chỉ tồn tại để phục vụ rải, nên sau khi ⑦ chốt thì nó được dựng rồi không ai
đọc — 0,73 giây mỗi lượt cho một dict chết, và tệ hơn là nó làm người đọc tưởng hệ có rải.
Cửa sổ đó vẫn cần để **chấm điểm**, nên nó sống ở `scripts/eval/*`.

Phân tích trần vẫn ở `src/submission/coverage.py`, nay là **mã phân tích, không nằm trên
đường chạy**: nó là tài liệu định lượng biện minh cho việc cắt dày hơn.

---

## Kết quả

Chấm bằng `scripts/eval/score_submission.py`, nguyên văn thể lệ:
`R@k = max_{i≤k} R(rᵢ)`, `k ∈ {1,5,20,50,100}`, `Final = (1/5)·Σ R@k`, ngân sách 100.

`L` = bề rộng cửa sổ đáp án `[s,e]` **do BTC quy định, ta không biết**. Thể lệ nói "thường
dưới 10 frame"; ví dụ trong PDF là `[500,510]` = 11 khung. Nên quét `L` thay vì chốt một số.

### Bộ 100 câu gán nhãn tay — đã dùng để chỉnh trọng số

| `L` | R@1 | R@5 | R@20 | R@50 | R@100 | **Final** |
|---|---|---|---|---|---|---|
| 9 | 0,0788 | 0,2896 | 0,5529 | 0,6596 | 0,6892 | **0,4540** |
| **11** | 0,1075 | 0,3187 | 0,5875 | 0,6983 | 0,7308 | **0,4886** |
| 21 | 0,1925 | 0,3883 | 0,6317 | 0,7550 | 0,7887 | **0,5512** |

### Bộ 110 câu GIỮ KÍN — chưa dùng để chọn tham số nào

| `L` | R@1 | R@5 | R@20 | R@50 | R@100 | **Final** |
|---|---|---|---|---|---|---|
| 9 | 0,0299 | 0,1318 | 0,2989 | 0,3667 | 0,4106 | **0,2476** |
| **11** | 0,0424 | 0,1481 | 0,3148 | 0,3852 | 0,4269 | **0,2635** |
| 21 | 0,0750 | 0,1773 | 0,3470 | 0,4091 | 0,4659 | **0,2948** |

Cả hai là **mô hình bi quan** — mốc ngữ nghĩa rơi ngẫu nhiên trong khe. Mô hình lạc quan
(mốc trùng keyframe) cho `L=11` = 0,6240 trên bộ tay.

⚠️ **Không so ngang hai bảng.** Câu ở bộ giữ kín dài **27 từ** so với **51 từ**, ít thông
tin hơn nên khó hơn. Chỉ so được **hiệu** giữa các cấu hình.

### Bộ 100 câu GT v2 — ground truth cấp shot do đội gán nhãn

*(`benchmark_ground_truth_final_v2`, chấm qua `representative_frame` để so ngang được.)*

**Cấu hình HIỆN TẠI — không rải** (`submission_now`), mô hình mốc lệch:

| `L` | R@1 | R@5 | R@20 | R@50 | R@100 | **Final** |
|---|---|---|---|---|---|---|
| 9 | 0,0450 | 0,0950 | 0,1008 | 0,1029 | 0,1037 | **0,0895** |
| **11** | 0,0546 | 0,1187 | 0,1254 | 0,1288 | 0,1300 | **0,1115** |
| 21 | 0,1154 | 0,2462 | 0,2650 | 0,2717 | 0,2742 | **0,2345** |

Bản **có rải** trước đây, cùng bộ, cùng hạt giống, để so:

| `L` | R@1 | R@5 | R@20 | R@50 | R@100 | **Final** |
|---|---|---|---|---|---|---|
| 9 | 0,0454 | 0,2104 | 0,5229 | 0,6142 | 0,6308 | **0,4047** |
| **11** | 0,0500 | 0,2508 | 0,5767 | 0,6763 | 0,6942 | **0,4496** |
| 21 | 0,1138 | 0,2838 | 0,5863 | 0,6842 | 0,6987 | **0,4733** |

🔴 **Xoá rải mất 0,3381 Final** ở `L=11` (0,1115 so với 0,4496; bắt cặp trên `L=9` cho
Δ=−0,3129, KTC95 [−0,3587, −0,2665], thắng 4 / thua 72). Đọc hai bảng theo **cột** thì thấy
cơ chế: `R@1` gần như y nhau (0,0546 so với 0,0500 — không rải còn hơn chút), rồi bản không
rải **đứng im** từ `R@5` trở đi. Tức **99 trong
100 suất ngân sách gần như vô giá trị** ở cấp khung: chúng là những keyframe cách nhau ~48
khung, mà chỉ một cửa sổ 11 khung được tính.

Rải và mật độ giải cùng một bài toán — phủ cái khe. Rải phủ nó **giả tạo** bằng ngân sách;
cắt dày phủ nó **thật** bằng dữ liệu. Nên xoá rải là đúng **với điều kiện** mật độ tăng.
Nếu thi trước khi cắt thêm khung, đây là khoản mất 0,31 có thật.

### Cùng bài nộp, chấm CẤP SHOT (`scripts/eval/score_shots.py`)

| bài nộp | R@1 | R@5 | R@20 | R@50 | R@100 | **Final** |
|---|---|---|---|---|---|---|
| không rải | **0,5600** | 0,7200 | 0,7400 | 0,7700 | 0,7700 | **0,7120** |
| có rải | 0,5600 | 0,5600 | 0,6800 | 0,7300 | 0,7300 | 0,6520 |

Hai thước **ngược chiều nhau**, và đó không phải nghịch lý — nó là câu hỏi `L`. Cấp shot
thì trúng khung nào trong shot cũng được, nên rải là **thuần lãng phí** suất; cấp khung với
`L=11` thì phải đúng khung, nên rải **thắng lớn**.

### 🔴 Con số quan trọng nhất: mất điểm KHÔNG phải do xếp hạng

`R@1` cấp shot = **0,5600** so với `R@1` cấp khung = **0,0533**. `R@100` thì hai thước gần
bằng nhau (0,77 so với 0,73). Đo trực tiếp trên 100 câu, `L=11`:

    shot ĐÚNG ở hạng 1                        56/100
    trong đó khung cũng vào cửa sổ ±5         6,1/56 = 10,9%
    ⟹ 50/100 câu mất điểm dù xếp hạng HOÀN HẢO
    khoảng cách khung nộp → mốc thật          trung vị 27 khung (p90 119)

Tách tiếp phần 56 câu đó:

    khung hạng 1 ĐÚNG là keyframe ground truth   3/56  =  5%
    khung hạng 1 là keyframe KHÁC trong shot    53/56  = 95%, lệch trung vị 5 khung

Nên chỗ hổng **không phải** "xếp hạng kém" mà là **độ phân giải thời gian**: hệ chỉ tay
được vào đúng shot và gần đúng keyframe (lệch trung vị 5 khung), nhưng cửa sổ đáp án 11
khung hẹp hơn khe keyframe 43 khung. Đây là lý lẽ định lượng cho việc cắt dày hơn, và nó
**loại** các hướng đắt hơn: tháp nhúng thứ hai, hợp điểm khác, tín hiệu mới — cả ba đều
tấn công xếp hạng, thứ đang ở 0,56 chứ không phải chỗ hỏng.

⚠️ Không ngoại suy bằng mô hình: mô hình hình học ước `26,3%` giữ được ở mật độ hiện tại
trong khi **đo được 10,9%** — lệch 2,4×. Muốn biết mật độ ×4 cho bao nhiêu thì phải cắt
thật rồi đo.

Độc lập ở mức: **0 câu trùng** bộ tay, **9/71 video trùng** (13%) — nhỏ nhưng có thật.

⚠️ GT này định danh theo **shot**, và README của nó ghi `representative_frame` *"không
phải đơn vị chấm chính"*. Chấm qua khung đại diện là **chặt hơn** ý định bộ GT; bản chấm
cấp shot ở `scripts/eval/score_shots.py`.

⚠️ Ánh xạ có bẫy: `shot_start`/`shot_end` (giây) của GT là **thời điểm keyframe đầu–cuối
trong shot**, KHÔNG phải biên shot. So chúng với biên metadata cho **0/105 khớp** và làm
tưởng dữ liệu lệch. Nối đúng là qua **`shot_id`**, khớp chính xác.

### Theo loại đề, `L = 11`

| loại | bộ tay | bộ giữ kín | **GT v2** |
|---|---|---|---|
| vision | 0,5158 | 0,3100 | **0,2800** ← thấp nhất |
| vision+ocr | **0,4846** ← thấp nhất | **0,2733** ← thấp nhất | 0,5360 |
| vision+asr | 0,6720 | 0,2467 | 0,5920 |
| **vision+ocr+asr** | **0,7733** | **0,4667** | **0,6720** |

Câu cần **cả ba nguồn** luôn cao nhất trên **cả ba bộ** — gấp 1,5 tới **2,4 lần** câu thuần
thị giác. Đó là phần dung hợp trả công, và nó là kết luận duy nhất bền qua ba bộ đề khác
hẳn nhau.

🔴 **Nhưng loại YẾU NHẤT thì đổi theo bộ eval.** Hai bộ đầu nói `vision+ocr`; GT v2 nói
`vision` thuần. Nên "chữa `vision+ocr`" mà tôi từng đặt làm việc #2 là **sai đề** — nó là
tính chất của bộ eval, không phải của hệ. Đề đúng: **nguồn thị giác đơn độc là chỗ yếu**,
và đó chính là thứ ensemble nhiều tháp nhúng tấn công.

---

## Bốn quyết định lớn, kiểm chéo trên bộ giữ kín

Bắt cặp theo truy vấn, bootstrap 4000 lần:

| quyết định | Δ | KTC95 | T/Th | |
|---|---|---|---|---|
| δ ASR `0,1064 → 0,2128` | **+0,0252** | [+0,0044, +0,0472] | 20/6 | ✓ |
| hợp điểm → **rổ ứng viên** | **+0,0153** | [+0,0063, +0,0256] | 12/1 | ✓ |
| bỏ VLM → **VLM cả rổ** | **+0,0356** | [+0,0091, +0,0659] | 18/7 | ✓ |

Cả bốn sống sót trên tập chưa từng dùng để chọn gì.

⚠️ Hai quyết định về **rải** (rải 7 hơn rải 1 `+0,1477`; thích ứng hơn cố định `+0,0280`)
cũng sống sót, nhưng **rải đã bị xoá** theo hướng cắt keyframe dày hơn — xem ⑦.

⚠️ VLM là **cú can thiệp mạnh và hiếm**, không phải bộ chỉnh êm: 85/110 câu nó không đụng
tới. Lợi ích đến từ **18 thắng / 7 thua** — tỉ lệ 2,6 ăn 1. Với đề 35 câu thì kỳ vọng chỉ
~8 câu bị đụng, nên `+0,0356` là kỳ vọng **dài hạn**.

### Ba con số giải thích kiến trúc

**Trần hình học 23,5%.** Keyframe cách nhau trung vị **48 khung**, cửa sổ đáp án ~10 khung.
Xác suất một cửa sổ chứa sẵn keyframe của ta chỉ 23,5% — trần cứng của nộp thuần keyframe,
bất kể truy xuất tốt đến đâu. Rải khung vào khe là cách vượt trần đó.

**Rổ đưa vào, reranker đẩy lên.** Trên 10 câu mà kiến trúc cũ trượt sạch: video đúng vào
được **4/10 → 8/10**, hạng **71–98 → 3–48**. Cần cả hai; mỗi cái một mình gần như vô dụng.

**Dư địa còn lại +0,2422.** `R@1 = 0,11` nhưng `R@100 = 0,73` — tìm được đáp án cho 73% số
câu nhưng chỉ đặt đúng hạng 1 được 11%. Rổ quyết định `R@100`; rerank chỉ dời cú trúng lên
sớm hơn. **Tiền còn lại nằm ở khâu xếp hạng, không ở khâu tìm.**

### Trọng số

```
③ modality (alpha)   visual 1/3 · ocr 1/3 · asr 1/3          ← ĐỀU
③ expansion (beta)   đều trong từng modality, 3 run ⟹ 1/3
⑤ giai đoạn          fused4 1,0 · crop 0,0 · vlm 0,25
```

Chỉ **tỉ lệ** có nghĩa — cả ba tầng đều cộng **đóng góp RRF**, nên trọng số diễn đạt
**mức tin cậy**, không sửa thang. Mặt mục tiêu **phẳng**: mỗi trọng số chỉ xác định được
trong khoảng rộng ~2,5 lần. Đừng chỉnh chữ số thứ ba; hãy mở rộng bộ eval.

🔴 **`vlm = 0,25` hết hiệu lực về căn cứ.** Nó quét ra khi ⑤ hợp bằng chuẩn hoá z, nơi
`fused4` và `vlm` cùng được đưa về `E=0, sd=1` trước khi cộng. Nay ⑤ hợp bằng RRF, nên
`0,25` là trọng số **tầng ba của RRF** — cùng đơn vị với `alpha` của ③, khác đơn vị với
thứ nó được đo. Đến khi quét lại, nó chỉ là **giá trị mang sang**.

Bảng trọng số 4 nguồn cũ (`visual 1,0 · object 0,0891 · ocr 0,1322 · asr 0,2128`) đã bỏ
cùng chuẩn hoá z. Nguồn `object` cũng đã gỡ khỏi đường chạy: [ĐO] bắt cặp trên 210 câu,
`ΔMRR +0,0064` KTC95 `[−0,0067, +0,0205]`, **151/210 câu không đổi gì** — bỏ là quyết
định về chi phí (1,3 GB `objects-full`), không phải về điểm.
⚠️ δ ASR đổi ×2 **không vì tinh chỉnh kỹ hơn** mà vì **hỗn hợp eval sai**: bộ cũ fit trên
326 câu gộp, trong đó 226 câu có **0%** cần ASR còn 100 câu có **81%**. Điểm hoà vốn là
`p > 61%`, với `p` = tỉ lệ đề thật cần OCR/ASR.

---

## Ý tưởng đã BÁC BỎ bằng đo

| ý tưởng | kết quả |
|---|---|
| RRF | −0,2927 |
| **Xếp hạng** bằng top riêng mỗi nguồn (vòng tròn) | −0,1430 — *dùng top riêng để **chọn ứng viên** thì GIỮ* |
| Bỏ dung hợp, thứ hạng thuần reranker | −0,045 |
| Bậc 1 mảnh cắt bật | −0,030; 88/100 lớp kiểm chéo chọn `crop=0` |
| Lấy top-K theo **shot** (cách NII-UIT) | −0,0841 — phép rải đã tự phủ shot |
| Khử trùng theo cảnh / theo video | −4,3pp / −12,0pp |
| Chuẩn hoá `rank` | 0,035 so với 0,460 |
| Query expansion đa biến thể | xấu hơn 17/30 câu |
| Đồng thuận giữa nguồn làm số hạng điểm | −0,0208 — mạnh gấp 48× nền nhưng vô dụng khi đã có điểm |
| Siết hạn ngạch mỗi video xuống 2–5 | −0,018 tới −0,046 |
| Phân bổ ngân sách nộp phi đồng đều | vùng phẳng, KTC chứa 0 |
| 1024 chiều thay 512 | Δ=−0,002, KTC chứa 0, tốn gấp đôi RAM |
| Token 2 âm tiết cho BM25 tiếng Việt | trung tính tới hơi tệ |
| Xếp hạng bằng HẠNG PHẦN TRĂM thay z-score | −0,033 — độ lớn điểm mang thông tin thật |
| Đưa **OCR** vào prompt ⑤c | −0,010; hại đúng loại cần OCR (−0,032 · −0,072). VLM có giá trị vì **độc lập**; đếm trùng nguồn nhiễu thì mất phán đoán thị giác. *ASR thì có lợi +0,040* |
| IDF trong-video cho OCR | lợi `vision+ocr` +0,007 nhưng giảm nửa `vision+asr`; tổng âm |
| VLM nhân thay vì cộng · VLM chỉ sửa 20 hạng đầu | −0,017 · −0,020 |

---

## QAFuse — nghiên cứu dung hợp hạng (E0–E4)

Nhánh nghiên cứu riêng, chấm trên `data/eval/bench_kis_gt.json` (100 câu) bằng **MRR** —
**không so ngang** với các bảng `Final` ở trên, vốn chấm theo luật BTC trên GT v2.
Kết quả ở `data/research/`, script ở `scripts/research/`.

### 🔴 Một file thiếu đã làm hỏng gần hết thang nghiên cứu

`load_frame_ms()` đọc `data/Framme/*/metadata/*.csv`. Thư mục đó bị xoá, hàm trả `{}`, và
`AsrSource` dùng bản đồ rỗng ấy để gắn segment vào khung — nên **ASR phủ 0/173.426 khung
và ghi 0 điểm cho MỌI truy vấn**. Không nổ, không cảnh báo; nó trông y hệt một nguồn đã
được đo và kết luận là vô dụng.

E1 quy điều đó cho câu truy vấn (*"có cụm 'Tìm video quay cảnh…' nên BM25 không khớp"*).
Lời giải thích ấy **không thể đúng** — không câu chữ nào khớp được với nguồn phủ 0 khung.

Đã vá: `load_frame_ms()` lùi về `data/OCR/ocr.jsonl`, phủ **173.426/173.426** khoá và
`pts_time` khớp `frame_idx/fps` ở **500/500** mẫu. Sau vá ASR phủ **96,9%**.

| | R@100 cũ | R@100 mới | MRR mới |
|---|---|---|---|
| asr only | 0,00 | **0,58** | 0,2058 |
| asr + `qa` | 0,00 | **0,62** | 0,3127 |
| visual+asr | 0,65 | **0,85** | 0,4596 |
| fusion all | 0,73 | **0,91** | 0,4751 |

**ASR là nguồn đơn mạnh thứ hai** theo R@100 (0,58 so với OCR 0,46), và expansion `qa`
nâng MRR **0,206 → 0,313**. Bản báo cáo cũ giữ ở `summary_ASR_DEAD.json` mỗi thư mục.

### E3 — ma trận cấu trúc phân cấp

| cấu hình | MRR | R@100 |
|---|---|---|
| V + O gốc + A gốc *(không QE)* | 0,4751 | 0,91 |
| V + O `qo` + A `qa` *(bỏ bản gốc)* | 0,4867 | 0,93 |
| V + (O gốc+`qo`)/2 + A gốc | 0,4989 | 0,92 |
| V + O gốc + (A gốc+`qa`)/2 | 0,4874 | 0,92 |
| **V + (O gốc+`qo`)/2 + (A gốc+`qa`)/2** ★ | **0,5281** | **0,93** |
| V + O gốc + O `qo` + A gốc + A `qa` *(PHẲNG)* | 0,4427 | 0,89 |

Ba kết luận, cả ba nằm trong mặc định của `hierarchical_rrf`:

- **phân cấp hơn phẳng 0,085 MRR** — khoản lớn nhất bảng. RRF phẳng cho modality nhiều
  expansion hơn nhiều quyền vote hơn chỉ vì đếm run;
- **giữ bản gốc bên cạnh expansion hơn bỏ nó 0,041** — expansion bổ sung, không thay thế;
- **QE cho cả hai nhánh văn bản** mới đạt đỉnh, cao hơn tổng hai phần riêng lẻ.

### E4 — Global Weighted RRF: **NO-GO**

Tune hai tầng (beta expansion → alpha modality) trên 210 câu tuning (gtv2 + holdout), báo
trên 100 câu held-out chưa đụng tới:

```
alpha* = (0,30 · 0,35 · 0,35)   ≈ đều
beta*  = (0,5 · 0,5) cho CẢ OCR lẫn ASR   ≈ đúng giá trị E3 đang dùng

held-out:  E3 đều MRR 0,5281  →  E4 có trọng số 0,4884
           ΔMRR bắt cặp −0,0398 · KTC95 [−0,0783, −0,0056] · thắng 10 / thua 18
```

Khoảng tin cậy nằm **trọn dưới 0**: trọng số toàn cục **kém hơn có ý nghĩa**, không phải
"không phân biệt được". Ngưỡng GO của playbook là +1 điểm %; đây là −4 điểm %.

⚠️ `queries_smoke` bị loại khỏi tập tuning: nó là **3 câu con của gtv2**, cùng `id`.

### Vì sao E4 hỏng lại là lý lẽ cho E5

Không phải "Equal RRF trừng phạt Visual" như E3 từng viết — mà là **một hằng số không
phục vụ được mọi loại đề**:

| loại đề | n | E3 | E4 | Δ |
|---|---|---|---|---|
| vision | 45 | 0,0966 | 0,0270 | **−0,0696** |
| vision+asr | 55 | 0,2576 | 0,2664 | +0,0088 |
| vision+ocr | 55 | 0,3519 | 0,3520 | +0,0001 |
| vision+ocr+asr | 55 | 0,4043 | 0,4310 | **+0,0266** |

Và cùng tín hiệu đó xuất hiện theo **kênh phát**, đo trên 310 câu có nhãn — modality
thắng cuộc đổi theo kênh, Visual đi từ **0,3919** (Báo Tuổi Trẻ) xuống **0,0126** (Báo
Thanh Niên), tức **31 lần**. Kênh lấy từ `data/media-info-aic25-b1.zip`, phủ 873/873 video.
Vì kho cố định, `alpha(kênh)` tính được **offline một lần**, không thêm gì vào ngân sách
2h30 — khác `alpha(truy vấn)` vốn cần phân loại mỗi câu lúc thi.

⚠️ Hai giả thuyết này **chưa tách được nhau**: nếu đề viết cho video ViVU tình cờ thiên
lời nói thì `alpha(kênh)` chỉ đang đo lại `alpha(loại đề)` qua một biến trung gian.

### Ý tưởng đã đo và BÁC BỎ ở nhánh này

| ý tưởng | kết quả |
|---|---|
| Query expansion cho **Visual** | dịch sang tiếng Anh hạ R@100 0,78 → 0,68; paraphrase tiếng Việt → 0,77. Jina-CLIP-v2 xử lý tiếng Việt gốc tốt hơn mọi bản viết lại |
| **Trọng số toàn cục** (E4) | −0,0398 MRR trên held-out, KTC trọn dưới 0 |
| **RRF phẳng** thay phân cấp | −0,085 MRR |
| **Title/metadata** làm nguồn xếp hạng | đứng riêng: video đúng vào top-10 chỉ **26/100**, MRR 0,0056. Thêm vào phân cấp với trọng số **đều thì HẠI** −0,0188; chỉ ×0,25 mới có lợi +0,0070 — dưới ngưỡng GO, và trọng số ấy lại chọn trên chính held-out nên là rò rỉ |

---

## Đối chiếu NII-UIT @ VBS2025

⚠️ VBS là hệ **có người trong vòng lặp** — người dùng chọn paraphrase, kéo trọng số, duyệt
shot. AIC sơ tuyển **nộp tự động**. Phần lớn thiết kế của họ không chuyển sang được.

| ý của họ | ta |
|---|---|
| Q&A: câu hỏi → câu mô tả trước khi truy xuất | **ĐÃ ÁP DỤNG** (`probe.declarativize`) |
| Hợp nhiều tháp nhúng (BEiT-3 · OpenCLIP · InternVL-G) | **chưa** — khoảng cách thật |
| Keyframe mỗi khung thứ 10 | ta khe 48 → trần 23,5%; ×4 dày lên 73,5% *(nhưng ở tiền xử lý)* |
| Xếp hạng theo shot | đã đo, **−0,0841** |
| Query expansion bằng LLM | **ĐÃ ÁP DỤNG, nhưng khác chỗ đặt** — xem ghi chú dưới |
| Temporal search **hai chiều** | **sai luật AIC** — TRAKE quy định chuỗi có thứ tự |
| Vật thể làm **bộ lọc cứng** | không còn bàn — nguồn vật thể đã gỡ khỏi đường chạy |

🔁 **Query expansion đã đảo kết luận, và chỗ đặt là lý do.** Bản đo cũ *thay* truy vấn
gốc bằng bản diễn giải rồi chấm một danh sách — mất bản gốc thì mất luôn phần nó đúng, và
ở VBS chính *người dùng* bù chỗ đó bằng cách chọn bản nào dùng. Nay bản mở rộng **không
thay** gì cả: mỗi bản là một lượt chấm riêng, bản gốc vẫn là một lượt, và tầng `beta` của
③ cộng chúng lại theo HẠNG. Không có ai chọn giúp, nên ta cộng đều thay vì chọn.

Họ **không có OCR và ASR**. Kho AIC là tin tức tiếng Việt — chữ dưới màn, tên kênh, lời
dẫn — nên hai nguồn đó là **lợi thế riêng của bài toán ta**.

---

## Chạy

Mỗi truy vấn là **một file `.txt`**, tên file thành `query_id`. Loại đề đoán theo: tên file
chứa `trake`/`qa`/`kis` → nội dung có `E1:`/`E2:` → còn lại là KIS.

```bash
modal run scripts/run.py --dir queries --out submission
```

```bash
modal run scripts/run.py --vlm-top-k 0    # bỏ ⑤c, tiết kiệm ~4 phút
modal run scripts/run.py --light          # chỉ thị giác, cho máy thiếu RAM
python scripts/eval/score_submission.py --sub submission --gt <gt.json>
modal run scripts/eval/make_queries.py    # sinh bộ eval giữ kín mới
```

**Đầu ra** `submission/{id}.csv`; cột thứ hai trở đi là **`frame_idx`** — số khung THẬT,
không phải `n`. Đo được 0/173.426 khung có hai giá trị bằng nhau.

Ngân sách: **~3,5 phút cho 35 câu** không bật ⑤c, **~7 phút** có bật (~5% của 2h30). Phí
cố định 80 giây nạp chỉ mục trả **mỗi lần gọi** — gom lô.

---

## Còn phải làm

| # | việc | vì sao |
|---|---|---|
| 1 | 🔴 **CẮT DÀY HƠN keyframe** | đo được: **50/100 câu mất điểm dù shot đúng ở hạng 1**. Khe keyframe 43 khung so với cửa sổ đáp án 11 khung. Đây là chỗ hổng LỚN NHẤT và là **chỗ duy nhất** đã chứng minh được là chỗ hổng |
| 2 | **Biết `p`** — tỉ lệ đề thật cần OCR/ASR | quyết trực tiếp δ ASR: `p > 61%` thì ×2 có lãi |
| 3 | Bộ eval **QA** và **TRAKE** | ⑥ chưa có phép đo nào; λ mới có 6 câu, p=0,69 |
| 4 | `writer.py` validator | chờ mẫu file nộp chính thức của BTC |

**Đã RÚT khỏi danh sách, kèm lý do:**

| việc cũ | vì sao rút |
|---|---|
| ~~Thu hẹp `R@1 ↔ R@100`~~ | **đặt sai đề.** Cấp shot `R@1` đã là 0,56 — xếp hạng không phải chỗ hỏng. Khoảng cách đó là **độ phân giải thời gian**, tức việc #1 mới |
| ~~Mạnh hoá nguồn thị giác đơn độc~~ | cùng lý do: khung hạng 1 chỉ lệch **trung vị 5 khung** so với keyframe ground truth. Thị giác đã tay được vào đúng chỗ |
| ~~Hợp nhiều tháp nhúng~~ | tấn công xếp hạng (0,56), không tấn công độ phân giải. ~$4,32/tháp + 0,36 GB để đổi lấy dư địa nhỏ hơn hẳn |
| ~~⑤b mảnh cắt~~ | đo tách nhóm màu: trống ở **cả** nhóm ≥2 màu. Bỏ |

---

## Bài học đo lường

**Đường ống KHÔNG tất định bit — nhưng sàn nhiễu đo được và nó nhỏ.** Hai lần chạy cùng
cấu hình khác nhau ở **7/110 câu** (`.map()` chia lô khác nhau ⟹ đệm khác ⟹ logit bfloat16
lệch chữ số cuối ⟹ lật thế hoà). Chênh lệch `Final` chỉ **0,0006** ở `L=11`, trong khi
hiệu ứng nhỏ nhất được báo là +0,0153 — **cách nhau 25 lần**. Mọi kết luận nằm trên sàn.

**Hai bộ chấm trong cùng một kho từng cài HAI mô hình cửa sổ khác nhau.**
`score_submission.py` cho mốc rơi đều trong **nửa khe keyframe**; `compare_arch.py` cho nó
rơi trong **nửa khe ∩ biên shot**. Cùng một bài nộp ra `0,2605` so với `0,2801` — lệch
0,0197, **gấp 3 lần** biên độ nhiễu hạt giống. Đã thống nhất về bản **có chặn biên shot**,
vì mốc ngữ nghĩa thuộc MỘT shot nên không thể rơi sang shot bên cạnh — đó là ràng buộc
thật, không phải lựa chọn mô hình. Mọi số tuyệt đối đã báo trước bản sửa đều là **cận
dưới**.

⚠️ Và biên độ nhiễu hạt giống thật là **0,0068** ở 32 lần gieo (lệch chuẩn 0,0017), co lại
theo `√(số lần gieo)`. Tôi từng đoán "~0,005" từ MỘT cặp quan sát — đo tử tế thì phải quét
nhiều hạt, không suy từ một lần.

**`hash()` của Python có muối ngẫu nhiên MỖI TIẾN TRÌNH — nó từng là hạt giống của bộ
chấm.** Ba script gieo `default_rng(abs(hash(qid)) % 2**31)`, gồm chính `score_submission.py`
đã sinh mọi bảng `R@k`. Hệ quả đo được: **cùng mã, cùng bản lưu, cùng ground truth mà Final
ra 0,0995 rồi 0,0948** — gấp 8 lần sàn nhiễu 0,0006, và đủ để **lật argmax** một phép quét
trọng số (`vlm=0,25` thành `vlm=0,125`). Δ bắt cặp trong cùng tiến trình vẫn đúng vì A và B
chung hạt; **số tuyệt đối thì không tái lập được**. Đã thay bằng `rscore.stable_seed`
(crc32) + 2 test, một test chạy ở tiến trình con để bắt đúng loại lỗi này. Bảng nào báo
trước bản sửa thì mang sai số ~0,005.

**Mô hình cửa sổ ôm keyframe ground truth là phép đo khép kín** — nó thưởng cho việc nộp
lại chính cái nhãn của mình. Quét phân bổ ngân sách theo mô hình đó cho "không rải" đứng
đầu với 0,7520; đo lại theo mô hình mốc lệch thì đúng cấu hình ấy tụt xuống **0,1424**.

**Lấy đỉnh trên chính tập đo là khớp quá, và lượng khớp quá đo được.** Kiểm chéo lồng 5
lớp: đỉnh biểu kiến +0,0241, kiểm chéo còn **+0,0175** — 27% mức lợi là ảo.

**Kiểm định phải chạy trên `Final`, không phải hạng thô.** Hạng đổi ở vùng sâu (200→400)
không đổi `R@k` nào. Đo trên hạng thô ra "36/2"; đo trên `Final` ra "11/1".

**Con số tự mâu thuẫn là dấu hiệu lỗi phép đo.** Mọi `R@k` tăng nhưng kiểm định ra "thua
36/60" — hoá ra nhãn thắng/thua bị đảo do `zip(B, A)`.

**`R@k` là CDF của hạng trúng, không phải điểm cộng dồn.** `R@k = max_{i≤k}` nên không
giảm theo `k` **theo định nghĩa**. Mỗi truy vấn chỉ nhận đúng 6 giá trị `Final`:
1 · 0,8 · 0,6 · 0,4 · 0,2 · 0, tuỳ hạng trúng.

---

## Thiên lệch phải biết

Cả hai bộ eval đều do **model viết câu khi đang nhìn khung đích**, nên điểm tuyệt đối là
**cận trên**. Bộ 100 câu còn do chính đội gán nhãn nên câu chữ hợp với hệ hơn đề người
ngoài. Bộ 110 câu giữ kín chống được **rò rỉ tham số** (0 video trùng, chưa dùng để chọn
gì) nhưng **không** chống được thiên lệch này.

Muốn số tuyệt đối đáng tin thì cần người **không nhìn khung** viết đề.

---

## Bố cục

```
scripts/run.py                 đường chạy tổng — ①②③④⑤⑥⑦
scripts/index/                 dựng lại chỉ mục, chạy theo số thứ tự
scripts/eval/
  score_submission.py          chấm ĐÚNG LUẬT BTC, quét L, bảng R@k
  compare_arch.py              so kiến trúc; suy spread/trọng số từ bản lưu ⑤
  make_queries.py              sinh bộ eval giữ kín, 4 chốt chống thiên lệch
src/
  ingestion/jina_encoder.py    Matryoshka: CẮT rồi mới chuẩn hoá
  ingestion/vector_index.py    chỉ mục phẳng, mang cả `n` lẫn `frame_idx`
  retrieval/probe.py           ①  tách mốc; Q&A → câu mô tả
  retrieval/sources.py         ②  bốn nguồn + `covered`
  retrieval/bm25.py                BM25, cố tình KHÔNG tự bỏ dấu
  retrieval/score_matrix.py    ③⑤ hierarchical_rrf — RRF hai tầng (alpha, beta)
  retrieval/pool.py            ⑤a rổ ứng viên: hợp top RIÊNG mỗi nguồn
  retrieval/dante.py           ④  DP O(N·T), trục mili-giây
  retrieval/rerank.py          ⑤bc mảnh cắt + VLM
  submission/kbest.py          ④⑦ k-best của cùng DP — KIS là N=1
  submission/coverage.py       ⑦  rải khung vào khe
  scoring/rscore.py                R-Score và Final, nguyên văn thể lệ
scripts/research/              QAFuse E0–E4 — công cụ THÍ NGHIỆM, ngoài đường chạy thi
  run_e1_e4_recheck.py         chạy lại thang E1–E3 sau khi sửa lỗi ASR
  run_e4_wrrf.py               tune hai tầng beta→alpha, kết luận NO-GO
data/research/                 kết quả E0–E4, kèm summary_ASR_DEAD.json (bản trước khi sửa)
```
