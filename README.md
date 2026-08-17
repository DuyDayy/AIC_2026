# Tầng truy vấn — AIC 2026

Hệ truy xuất video đa phương thức cho vòng sơ tuyển. Tiền xử lý (khung hình · ASR · OCR ·
vật thể) đã xong; kho này là **tầng truy vấn**: từ đề bài ra bài nộp.

Vận hành ngày thi: [`RUNBOOK.md`](RUNBOOK.md) · Kế hoạch tối ưu: [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md)

```
⓪ mã hoá offline   173.426 khung → lưu 1024 chiều, tra ở 512
       │
đề → ① probe → ② bốn nguồn ─┬─→ ③ dung hợp có trọng số ─────┐
                            │                               ├→ ⑤c VLM → ④ kbest → ⑦ nộp
                            └─→ ⑤a RỔ: top RIÊNG mỗi nguồn ─┘
```

**Rổ chọn ai vào vòng trong; dung hợp + VLM quyết thứ hạng.**

---

## Công nghệ

| tầng | công nghệ | vai trò |
|---|---|---|
| ⓪ | **jina-clip-v2** (0,9B, 89 ngôn ngữ) | nhúng chung ảnh–chữ; Matryoshka lưu 1024, tra 512 |
| ① | tách mốc `E1:`/`E2:`, rút trích dẫn | một truy vấn → N probe. Q&A: câu hỏi → câu **mô tả** |
| ② | jina-clip · **BM25** ×3 | thị giác · OCR · vật thể · lời nói, mỗi nguồn chấm độc lập |
| ③ | chuẩn hoá **z trên tập có dữ liệu** | hợp 4 nguồn; khung không phủ nhận đúng kỳ vọng, không nhận 0 |
| ⑤a | **rổ ứng viên** (`pool.py`) | mỗi nguồn đề cử top 40 riêng → rổ ~155 khung, **bộ lọc cứng** |
| ⑤b | mảnh cắt vật thể + jina-clip | **TẮT** — đo được nó phá điểm |
| ⑤c | **Qwen2.5-VL-7B** | `P(khung khớp)` = softmax trên đúng hai token `1`/`0` |
| ④ | **DANTE** k-best (`kbest.py`) | DP thứ tự thời gian `O(N·T)`; **KIS = TRAKE với N=1** |
| ⑥ | Qwen2.5-VL | chỉ đề Q&A: đọc khung + OCR + lời nói → sinh `answer` |
| ⑦ | một khung mỗi mốc, 100 mốc | rải đã XOÁ — chờ bộ keyframe dày hơn nâng trần 23,5% |

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

### ② Bốn nguồn điểm

| nguồn | công nghệ | dữ liệu | phủ |
|---|---|---|---|
| thị giác | jina-clip-v2, cosine | vector 512 chiều | 100% |
| OCR | **BM25** (`k1=0,6 · b=0,9`) | chữ hiện trên màn | 98% |
| vật thể | **BM25** (mặc định) | nhãn vật thể đã phát hiện | 99% |
| lời nói | **BM25** (`k1=1,5 · b=0,0`) | ASR trong cửa sổ **±5 giây** quanh khung | 97% |

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

⚠️ Nguồn **vật thể có MRR 0,0003**, hạng đáp án cỡ 3.000. Chỉnh tham số cho nó là vô
nghĩa; vấn đề ở chất lượng nhãn.

### ③ Ma trận S — dung hợp

| | |
|---|---|
| phép chuẩn hoá | **z trên riêng tập `covered`** ⟹ `E[z \| covered] = 0` |
| trọng số | `visual 1,0 · object 0,0891 · ocr 0,1322 · asr 0,2128` |
| cách rút | bootstrap 200 lần + kiểm chéo lồng 5 lớp |
| đã bác bỏ | RRF (−0,2927) · chuẩn hoá `rank` (0,035 so với 0,460) |

Đây là chỗ RRF chết. Nguồn phủ một phần, nên nếu coi khung không có dữ liệu là **điểm 0**
thì *việc được chấm* tự thành lợi thế bất kể liên quan. Chuẩn hoá z trên riêng tập được phủ
khiến khung không phủ nhận đúng **kỳ vọng**. `rank` hỏng vì trải đều thứ hạng phá mất sức
phân biệt của nguồn thị giác vốn dày đặc.

### ⑤a Rổ ứng viên — `src/retrieval/pool.py`

| | |
|---|---|
| cách chọn | mỗi nguồn đề cử **top 40 của riêng nó**, hợp lại |
| kích thước | ~155 khung/truy vấn (`POOL_CAP = 200`) |
| vai trò | **bộ lọc CỨNG** — chỉ khung trong rổ mới được nộp |
| chốt 1 | `score > 0` — khung có chữ nhưng không khớp từ nào thì không được đề cử |
| chốt 2 | **10 khung/video/nguồn** — chặn nguồn cho điểm phẳng cả video |
| lai lịch | `provenance` ghi nguồn nào đề cử khung nào |

Thị giác mang hệ số `1,0` còn ba nguồn kia `0,09–0,21`, nên khung chỉ có bằng chứng **thuần
OCR** gần như không nổi lên trong điểm đã hợp: đo được **10/100 câu** có video đúng vắng mặt
hoàn toàn, mà 9/10 nằm ở hạng 16–62 — với tới được, chỉ là không được với tới.

Chốt 2 sinh ra từ một ca thật: OCR khớp **logo kênh** hiện suốt video ⟹ điểm phẳng ⟹ top-40
của nó là 40 khung tuỳ tiện cùng một video.

### ⑤b Bậc 1 — mảnh cắt vật thể ⚠️ TẮT

| | |
|---|---|
| công nghệ | cắt bbox vật thể → mã hoá bằng jina-clip → cosine với truy vấn |
| sinh ra để chữa | *attribute binding* — đảo màu giữa hai vật chỉ đổi vector chữ **0,026** |
| lợi trên đề màu | 54% → **77%** đúng màu (8 câu thuần màu, chấm bằng mắt) |
| **trên bộ 100 câu** | **−0,030** · 88/100 lớp kiểm chéo chọn `crop = 0` |
| trạng thái | **TẮT** (`RERANK_WEIGHTS["crop"] = 0`) — trọng số 0 ⟹ không tính luôn |

Tháp chữ mã hoá `{áo, xe, đỏ, trắng}` như **túi khái niệm**; cắt vật ra mã hoá riêng buộc
màu vào vật bằng cấu trúc. Hai phép đo **không mâu thuẫn** — bộ 100 câu gần như không có
câu phụ thuộc màu. Bật lại bằng cách đặt `crop > 0`.

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

🔴 **Phép rải khung đã bị XOÁ**, và điều kiện để việc đó đúng phải nói rõ. Ở mật độ
keyframe hiện tại, rải mua rất nhiều điểm:

    [ĐO] bộ giữ kín 110 câu:  không rải → 0,0857  ·  rải thích ứng → 0,2334
    tức bỏ rải mất ~63% điểm

Nguyên nhân là **hình học**: keyframe cách nhau trung vị **48 khung** còn cửa sổ đáp án
thể lệ ~10 khung, nên xác suất một cửa sổ chứa sẵn keyframe của ta chỉ **23,5%**.

Trần đó là **hàm của mật độ**, và kế hoạch cắt dày hơn hoá giải đúng nó:

| khe keyframe | trần `L=9` | trần `L=11` |
|---|---|---|
| 48 (hiện tại) | 23,5% | 28,6% |
| 24 (×2 dày) | 45,7% | 53,1% |
| **12 (×4 dày)** | **73,5%** | **81,1%** |
| 10 (mỗi khung thứ 10) | 90,0% | 100% |

Ở ×4 dày, mỗi keyframe tự phủ cửa sổ của nó và rải thành vô nghĩa. **Xoá rải là đúng SAU
KHI bộ keyframe dày hơn có thật** — trước đó thì nó là mất 63% điểm.

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

| `L` | R@1 | R@5 | R@20 | R@50 | R@100 | **Final** |
|---|---|---|---|---|---|---|
| 9 | 0,0471 | 0,2104 | 0,5163 | 0,6067 | 0,6242 | **0,4009** |
| **11** | 0,0542 | 0,2425 | 0,5596 | 0,6646 | 0,6833 | **0,4408** |
| 21 | 0,1013 | 0,2692 | 0,5800 | 0,6792 | 0,6900 | **0,4639** |

Đây là **lần đầu đủ cấu hình cuối chạy cùng nhau** — rải thích ứng · `BM25_PARAMS` riêng
nguồn · `VLM_TOP_K=160`. Trước đó ba Δ chỉ được đo riêng lẻ.

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
dung hợp 4 nguồn   visual 1,0 · object 0,0891 · ocr 0,1322 · asr 0,2128
sau rerank         fused4 1,0 · crop 0,0     · vlm 0,25
```

Chỉ **tỉ lệ** có nghĩa — cả ba đi qua cùng phép chuẩn hoá z nên trọng số chỉ diễn đạt
**mức tin cậy**, không sửa thang. Mặt mục tiêu **phẳng**: mỗi trọng số chỉ xác định được
trong khoảng rộng ~2,5 lần. Đừng chỉnh chữ số thứ ba; hãy mở rộng bộ eval.

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

## Đối chiếu NII-UIT @ VBS2025

⚠️ VBS là hệ **có người trong vòng lặp** — người dùng chọn paraphrase, kéo trọng số, duyệt
shot. AIC sơ tuyển **nộp tự động**. Phần lớn thiết kế của họ không chuyển sang được.

| ý của họ | ta |
|---|---|
| Q&A: câu hỏi → câu mô tả trước khi truy xuất | **ĐÃ ÁP DỤNG** (`probe.declarativize`) |
| Hợp nhiều tháp nhúng (BEiT-3 · OpenCLIP · InternVL-G) | **chưa** — khoảng cách thật |
| Keyframe mỗi khung thứ 10 | ta khe 48 → trần 23,5%; ×4 dày lên 73,5% *(nhưng ở tiền xử lý)* |
| Xếp hạng theo shot | đã đo, **−0,0841** |
| Query expansion bằng LLM | đã đo **xấu hơn** — ở VBS *người* chọn bản diễn giải |
| Temporal search **hai chiều** | **sai luật AIC** — TRAKE quy định chuỗi có thứ tự |
| Vật thể làm **bộ lọc cứng** | rủi ro — nguồn vật thể của ta yếu nhất |

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
| 1 | 🔴 **Thu hẹp `R@1 ↔ R@100`** | dư địa **+0,2422**. Đã loại trừ: hạng phần trăm (−0,033), nhân thay cộng (−0,017), VLM chỉ sửa 20 đầu (−0,020), phá hoà đồng thuận (0 câu đổi). **Cách hợp điểm không phải chỗ** — phải thêm tín hiệu mới |
| 2 | 🔴 **Mạnh hoá NGUỒN THỊ GIÁC đơn độc** | loại yếu nhất đổi theo bộ eval, nhưng câu **không có nguồn văn bản nào đỡ** luôn ở nhóm cuối. Trùng với việc #4 |
| 3 | **Biết `p`** — tỉ lệ đề thật cần OCR/ASR | quyết trực tiếp δ ASR: `p > 61%` thì ×2 có lãi |
| 4 | Hợp nhiều tháp nhúng | đánh đúng dư địa #1; ~$4,32/tháp + thời gian thi |
| 5 | Bộ eval **QA** và **TRAKE** | ⑥ chưa có phép đo nào; λ mới có 6 câu, p=0,69 |
| 6 | `writer.py` validator | chờ mẫu file nộp chính thức của BTC |

---

## Bài học đo lường

**Đường ống KHÔNG tất định bit — nhưng sàn nhiễu đo được và nó nhỏ.** Hai lần chạy cùng
cấu hình khác nhau ở **7/110 câu** (`.map()` chia lô khác nhau ⟹ đệm khác ⟹ logit bfloat16
lệch chữ số cuối ⟹ lật thế hoà). Chênh lệch `Final` chỉ **0,0006** ở `L=11`, trong khi
hiệu ứng nhỏ nhất được báo là +0,0153 — **cách nhau 25 lần**. Mọi kết luận nằm trên sàn.

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
  retrieval/score_matrix.py    ③  chuẩn hoá z trong tập có dữ liệu
  retrieval/pool.py            ⑤a rổ ứng viên: hợp top RIÊNG mỗi nguồn
  retrieval/dante.py           ④  DP O(N·T), trục mili-giây
  retrieval/rerank.py          ⑤bc mảnh cắt + VLM
  submission/kbest.py          ④⑦ k-best của cùng DP — KIS là N=1
  submission/coverage.py       ⑦  rải khung vào khe
  scoring/rscore.py                R-Score và Final, nguyên văn thể lệ
```
