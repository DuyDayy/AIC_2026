# Tầng truy vấn — AIC 2026

Hệ truy xuất video đa phương thức cho vòng sơ tuyển. Tiền xử lý (khung hình · ASR · OCR ·
vật thể) đã xong; kho này là **tầng truy vấn**: từ đề bài ra bài nộp.

```
⓪ mã hoá offline  173.426 khung → lưu 1024 chiều, tra ở 512
      │
đề → ① probe → ② bốn nguồn ─┬─→ ③ ma trận S ──────┐
                            │                      ├→ ⑤b mảnh cắt → ⑤c VLM → ④ DANTE ─┬→ ⑦ nộp
                            └─→ ⑤a RỔ ỨNG VIÊN ────┘                                   │
                                 hợp top riêng                          QA → ⑥ đầu đọc ─┘
```

**Rổ chọn ai vào vòng trong; reranker quyết thứ hạng.** ② chấm bốn nguồn độc lập
(thị giác · vật thể · OCR · ASR). Rồi hai đường tách ra:

- **⑤a rổ ứng viên** — mỗi nguồn tự đề cử **top 40 của riêng nó**, hợp lại thành rổ ~155
  khung. Nhờ vậy một khung chỉ có bằng chứng **thuần OCR** vẫn vào được vòng trong, dù
  điểm hợp của nó thấp. Rổ là **bộ lọc cứng**: chỉ khung trong rổ mới được nộp.
- **③ ma trận S** — chuẩn hoá z rồi cộng có trọng số, dùng làm điểm nền phá hoà và làm
  đầu vào cho ④ DANTE.

⑤ chấm lại rổ theo **thác nước hai bậc**: ⑤b cắt vật thể ra mã hoá bằng jina-clip (rẻ,
quét cả rổ), ⑤c cho VLM chấm `P(khớp)` trên **top 30 sau bậc 1** (đắt, chỉ chấm phần đầu).
Thứ hạng cuối là `0,30·nền + 1,0·mảnh cắt + 1,0·VLM`.

⚠️ Xếp hạng **bằng** top riêng của từng nguồn (xen kẽ vòng tròn) đã thử và **bác bỏ**:
−0,1430 Final. Rổ và xếp hạng là hai câu hỏi khác nhau — xem ③.

---

## Chạy

Thả file `.txt` vào `queries/`, mỗi truy vấn một file. Tên file thành `query_id`.

```bash
modal run scripts/run.py                    # ./queries → ./submission
modal run scripts/run.py --spread 7         # 🔴 RẢI khung vào khe — xem ⑦, +3,3 lần điểm
modal run scripts/run.py --no-rerank        # bỏ cả ⑤b và ⑤c
modal run scripts/run.py --vlm-top-k 0      # giữ ⑤b, bỏ ⑤c (VLM)
modal run scripts/run.py --light            # chỉ thị giác, cho máy thiếu RAM
```

🔴 **Mặc định `--spread 1` nộp thuần keyframe, và nó mất 3,3 lần điểm** (0,4477 → 0,1366).
Xem ⑦ để biết vì sao đó là trần **hình học** chứ không phải vấn đề xếp hạng.

Sau khi chạy, so kiến trúc và suy các `spread` khác **không tốn GPU**:

```bash
python scripts/eval/compare_arch.py --new submission_new --old submission_bench
python scripts/eval/score_submission.py --sub submission_new    # chấm đúng luật BTC
```

Loại đề đoán theo thứ tự: tên file chứa `trake`/`qa`/`kis` → nội dung có `E1:`/`E2:` →
còn lại là KIS.

**Đầu ra** `submission/{id}.csv`:

```
L22_V028,24590            KIS   — video_id, frame_idx
L26_V456,7256,Ba          QA    — thêm answer
L26_V082,4412,5564,6210   TRAKE — N frame_idx tăng dần
```

Cột thứ hai trở đi là **`frame_idx`** (số khung thật), không phải `n` (số thứ tự
keyframe). Đo được **0/173.426** khung có hai giá trị bằng nhau, lệch trung vị 5.267.

Chỉ tháp văn bản và mã hoá mảnh cắt cần GPU. Mọi thứ còn lại chạy tại máy, **$0**. Vector
truy vấn cache theo băm nội dung ở `data/embed/query_cache.npz` — chạy lại cùng bộ truy
vấn thì không tốn GPU lần nào nữa.

### Dựng lại chỉ mục

```bash
modal run scripts/index/2_encode_frames.py --benchmark 400   # ĐO trước
modal run scripts/index/2_encode_frames.py                   # lượt đủ, ~$4,32
python  scripts/index/3_build_index.py --src /tmp/emb/embed-jina-v2
modal run scripts/index/4_write_manifest.py                 # provenance
```

---

## Kết quả — chấm ĐÚNG LUẬT BTC trên đầu ra thật

100 truy vấn gán nhãn tay (`export_for_fusion/benchmark_queries.json`), chạy đủ ①→⑦, chấm
bằng `scripts/eval/score_submission.py`: `R@k = max_{i≤k} R(rᵢ)`, `k ∈ {1,5,20,50,100}`,
`Final = (1/5)·Σ R@k`, ngân sách 100 đáp án, `--spread 7`.

### Bộ mới — R@1 … R@100 và Final

**Mô hình BI QUAN** (mốc ngữ nghĩa rơi ngẫu nhiên trong khe — dùng cho mọi so sánh):

| `L` | R@1 | R@5 | R@20 | R@50 | R@100 | **Final** |
|---|---|---|---|---|---|---|
| 9 | 0,0788 | 0,2896 | 0,5529 | 0,6596 | 0,6892 | **0,4540** |
| **11** *(ví dụ thể lệ)* | 0,1075 | 0,3187 | 0,5875 | 0,6983 | 0,7308 | **0,4886** |
| 21 | 0,1925 | 0,3883 | 0,6317 | 0,7550 | 0,7887 | **0,5512** |

**Mô hình LẠC QUAN** (mốc trùng keyframe của ta):

| `L` | R@1 | R@5 | R@20 | R@50 | R@100 | **Final** |
|---|---|---|---|---|---|---|
| 9 | 0,4400 | 0,4400 | 0,6600 | 0,7800 | 0,8000 | **0,6240** |
| **11** | 0,4400 | 0,4400 | 0,6600 | 0,7800 | 0,8000 | **0,6240** |
| 21 | 0,4400 | 0,4600 | 0,6700 | 0,7900 | 0,8200 | **0,6360** |

**Hai bảng vì `[s,e]` là tham số duy nhất ta không biết.** Bảng lạc quan giả định mốc ngữ
nghĩa trùng keyframe của ta — nó **khép kín**, tự thưởng cho việc nộp lại chính cái nhãn
của mình. Bảng bi quan giả định mốc rơi đều trong khe. Sự thật nằm giữa; gộp thành một số
là tự lừa mình.

### So với bản cũ

| `L` | | cũ | **mới** | Δ |
|---|---|---|---|---|
| 9 | lạc quan | 0,5540 | **0,6240** | **+0,0700** |
| 9 | bi quan | 0,4246 | **0,4540** | **+0,0294** |
| 11 | lạc quan | 0,5540 | **0,6240** | **+0,0700** |
| 11 | bi quan | 0,4412 | **0,4886** | **+0,0474** |
| 21 | lạc quan | 0,5660 | **0,6360** | **+0,0700** |
| 21 | bi quan | 0,4956 | **0,5512** | **+0,0556** |

Dương ở **mọi `L`** và **cả hai** mô hình — không phải một điểm may mắn.

### Theo loại đề (`L = 11`, mô hình LẠC QUAN)

| loại | n | cũ | **mới** | Δ |
|---|---|---|---|---|
| vision | 19 | 0,4737 | **0,5158** | +0,0421 |
| vision+ocr | 26 | 0,4615 | 0,4846 | +0,0231 |
| vision+asr | 25 | 0,5600 | **0,6720** | **+0,1120** |
| **vision+ocr+asr** | 30 | 0,6800 | **0,7733** | **+0,0933** |

Mức lợi dồn đúng vào **câu cần lời nói** (+0,112) — nhất quán với việc δ ASR vừa tăng ×2.
Câu cần cả ba nguồn ăn điểm gấp **1,50×** câu thuần thị giác.

⚠️ Bảng này là **mô hình lạc quan** — `score_submission.py` tách theo loại trên cột đó.
Kiểm được: trung bình có trọng số `(19·0,5158 + 26·0,4846 + 25·0,6720 + 30·0,7733)/100
= 0,6240`, khớp đúng ô `L=11` lạc quan ở bảng trên.

### Đọc bảng `R@k` cho đúng — nó là CDF của hạng trúng, không phải điểm cộng dồn

`R@k = max_{i≤k} R(rᵢ)` là **max trên tiền tố dài `k`**, nên nó **không giảm** theo `k`
theo định nghĩa. Trúng ở hạng 1 thì cả năm mốc bằng 1 **cùng lúc** — không phải "tính
tiếp lần nữa". Với một truy vấn, `Final` chỉ nhận **đúng 6 giá trị**:

| hạng trúng | 1 | 2–5 | 6–20 | 21–50 | 51–100 | trượt |
|---|---|---|---|---|---|---|
| **Final** | **1,0** | 0,8 | 0,6 | 0,4 | 0,2 | 0,0 |

Nên hàng `R@k` trong bảng trên chính là **phân phối tích luỹ của hạng trúng**: `R@1 =
0,1075` nghĩa là 10,75% truy vấn trúng ngay hạng 1; `R@100 = 0,7308` nghĩa là 73,08%
trúng ở đâu đó trong 100 ô. `Final` là trung bình CDF tại 5 điểm — tức thước đo **trúng
SỚM tới mức nào**.

### 🔴 Trần của mọi phép rerank là `R@100` — dư địa +0,2422

Hệ quả trực tiếp: **rổ ứng viên quyết định `R@100`; rerank chỉ dời hạng trúng lên sớm hơn
BÊN TRONG rổ đó.** Nếu mọi cú trúng hiện có đều dời được về hạng 1 thì cả năm `R@k` bằng
`R@100`:

| | Final (`L=11`, bi quan) |
|---|---|
| hiện tại | 0,4886 |
| **trần nếu rerank hoàn hảo** | **0,7308** |
| **dư địa** | **+0,2422** |

Lớn hơn tổng mọi khoản tinh chỉnh trọng số trong trang này (+0,0175 và +0,0149) **gấp
bảy lần**. Đó là chỗ đáng đầu tư tiếp — không phải vặn thêm chữ số cho trọng số.

Muốn nâng `R@100` thì phải mở rộng rổ (⑤a) hoặc sửa nguồn; muốn thu hẹp khoảng cách
`R@1 ↔ R@100` thì phải làm reranker mạnh hơn.

### Hình dạng đường cong nói hệ hỏng ở đâu

`R@1 = 0,11` nhưng `R@100 = 0,73` (bi quan, `L=11`). Hệ **yếu ở xếp hạng đầu bảng, mạnh ở
recall** — mức tăng lớn nhất nằm ở `R@5 → R@20` (+0,269). Đó là chỗ còn tiền: đưa được
đáp án vào top 100 rồi, việc còn lại là đẩy nó lên top 5.

⚠️ Con số nên tin cho phần **cải thiện** là **+0,0175** (kiểm chéo lồng nhau), không phải
chênh lệch thô ở bảng trên — bảng trên không kiểm chéo. Xem ③ và ⑤.

## Ngân sách thời gian — kỳ thi chỉ có 2 giờ 30

"Chạy đúng" chưa đủ; đường ống phải **xong trong giờ thi**. Kiến trúc rổ + rerank 2 bậc
làm mỗi truy vấn nặng hơn (rổ ~156 khung thay vì top-100, ~304 mảnh cắt mỗi câu), nên
phải đo chứ không ước.

**Nút cổ chai KHÔNG phải GPU mà là ĐƯỜNG TRUYỀN.** Bậc 1 từng gọi `.remote()` 76 lượt
tuần tự, mỗi lượt tải ~10 MB base64, GPU rảnh gần hết thời gian:

| | `.remote()` tuần tự | **`.map()` song song** |
|---|---|---|
| tốc độ mã hoá | 2,7 mảnh/giây | **85–160 mảnh/giây** |
| 30.397 mảnh (100 câu) | ~3 giờ | **~7 phút** |
| container | 1 | 7 |

Không đổi một dòng logic nào — chỉ đổi cách gọi. Cùng cách chữa áp cho ⑤c (VLM).

⚠️ `.map()` giữ nguyên thứ tự đầu vào, mà `vspan` đánh chỉ số theo **vị trí** trong danh
sách việc. Lệch một là gán điểm VLM sang truy vấn khác **trong im lặng** — nên có một
phép kiểm độ dài, sai thì dừng hẳn chứ không chạy tiếp.

`run.py` in đồng hồ từng tầng (`⏱ ②③ chấm 4 nguồn`, `⑤a dựng rổ`, `⑤b cắt mảnh`,
`⑤b mã hoá + hợp`) để lúc thiếu giờ biết **cắt đúng chỗ**, không cắt mò.

**Cắt giờ theo thứ tự này:**

| thiếu giờ thì bỏ | lệnh | mất gì |
|---|---|---|
| ⑤c bậc 2 (VLM) | `--vlm-top-k 0` | chỉ còn bậc 1 |
| ⑤ cả hai bậc | `--no-rerank` | mất phần đẩy hạng — nhóm A tụt lại đáy |
| ba nguồn văn bản | `--light` | mất dung hợp, chỉ còn thị giác |

⚠️ Chi phí cố định **39 giây nạp chỉ mục + 4 nguồn cho MỖI lần gọi `modal run`**. Gọi
từng câu một là trả phí đó mỗi câu — **gom lô** rồi chạy một lần.

---

## Bốn loại đề, hai engine

| đề | probe từ | N | engine | dòng nộp |
|---|---|---|---|---|
| Textual KIS | tháp văn bản | 1 | ④ | `video_id, frame_id` |
| Video KIS | tháp ảnh | 1 | ④ | `video_id, frame_id` |
| TRAKE | tháp văn bản | N | ④ | `video_id, frame_ids[N]` |
| QA | tháp văn bản | 1 | ④ + ⑥ | `video_id, frame_id, answer` |

Ba đề đầu là **cùng một bài toán**: tìm video `v` và bộ khung tăng dần `t₁ < … < t_N`
cực đại tổng điểm. `N = 1` làm hệ thức truy hồi suy biến thành `max_t S[0,t]` — tức phép
max-pool thường. Không cần nhánh code riêng.

QA tách ra **đúng một chỗ**: phải trả thêm chuỗi `answer`.

---

## ⓪ Mã hoá — Jina CLIP v2

| | |
|---|---|
| chỉ mục | **173.426 × 1024** · 873/873 video · thiếu 0 khung |
| đĩa | 357,7 MB (fp16) |
| tra cứu | **512 chiều**, ~6 ms/truy vấn · quét phẳng chính xác, không ANN |
| chi phí | **$4,32** · 24 phút · A10G × 10 container · đo được 10,93 khung/s/container |

Lưu **đủ 1024**, cắt Matryoshka **lúc đọc**. Giá trị của Matryoshka là *quyền chọn số
chiều sau*, không phải bản thân phép cắt — cắt lúc mã hoá là vứt quyền đó để tiết kiệm
178 MB đĩa.

**Chiều tra cứu = 512**, đo trên 226 truy vấn:

| chiều | R@1 | R@10 | R@100 | Final | thời gian |
|---|---|---|---|---|---|
| 1024 | 0,155 | 0,407 | 0,681 | 0,4558 | 11 ms |
| **512** | 0,142 | **0,407** | **0,690** | **0,4566** | **6 ms** |
| 256 | 0,146 | 0,403 | 0,699 | 0,4496 | 3 ms |
| 128 | 0,142 | 0,350 | 0,646 | 0,4230 | 2 ms |

512 bằng hoặc hơn 1024 trên mọi chỉ số tổng hợp và **nhanh gấp đôi**. 128 thì sập — đáy
nằm giữa 256 và 128, không phải "càng nhỏ càng tốt".

Provenance đầy đủ ở `artifacts/embed/embed/manifest.json`: commit SHA của model và của
remote-code (hai giá trị **khác nhau** — lớp xử lý ảnh do remote code định nghĩa),
preprocessing, phiên bản gói, hash artefact.

---

## ① Probe hoá

Đúng **hai việc**, cả hai xác định: không model, không tham số, không ngẫu nhiên.

- **Tách N mốc** — `E1:`/`E2:` tường minh trước, rồi từ nối **ở đầu câu**
- **Rút chuỗi trong ngoặc kép** — 14,4% đề có

Văn bản đưa cho encoder là **nguyên văn** đoạn của mốc: không rút gọn, không viết lại.

**Tách mốc phải bảo thủ.** Đo trên 30 đề thật:

| từ nối | đầu câu | giữa câu | dùng làm ranh giới? |
|---|---|---|---|
| sau đó · tiếp đó · cuối cùng | 9 | 1 | ✅ |
| **lần lượt** | 0 | **2** | ❌ không bao giờ |
| **đầu tiên** | 0 | **5** | ❌ không bao giờ |

> q12 *"các người mẫu **lần lượt** bước trên sàn"* — một cảnh, không phải chuỗi

Tách nhầm chẻ một cảnh liền mạch thành hai mốc rồi ép DP tìm hai khoảnh khắc cách nhau —
và **không gì báo**. Tách **thừa** nguy hiểm hơn tách **thiếu**.

---

## ② Bốn nguồn điểm

| nguồn | công dụng | độ phủ |
|---|---|---|
| cosine jina-clip-v2 | khớp ngữ nghĩa hình ảnh | 100% |
| BM25 trên OCR | chữ hiện trên khung | 97,7% |
| BM25 trên `objects-full` | vị trí · số lượng · lớp | 100% |
| BM25 trên ASR | lời nói, cửa sổ ±5 giây | 96,9% |

**`covered` không phải tiện ích — nó là điều kiện để ③ đúng.** Điểm 0 có hai nghĩa khác
hẳn nhau: *"có dữ liệu và không khớp"* so với *"không có dữ liệu"*. Gộp chúng là lỗi đã
giết RRF.

⚠️ **Bỏ dấu cả hai phía.** `bm25.py` cố tình không tự bỏ dấu — nó ép chỗ gọi dùng
`src/text/fold.py`, cài đặt **duy nhất** trong toàn hệ. Đưa truy vấn còn dấu vào chỉ mục đã bỏ dấu làm `object` chấm ra
trung vị hạng **14.407** thay vì 3.653 — `looks_unfolded` bắt được lỗi này.

**Cửa sổ ASR = 5.000 ms**, và hoá ra gần như không phải tham số: độ phủ đã **93% ngay ở
cửa sổ 0** vì đoạn Whisper dài hàng chục giây. Nới lên 45 giây chỉ thêm 5 điểm phần trăm.

---

## ③ Ma trận S

```
S[i,t] = α·z_visual + β·z_object + γ·z_ocr + δ·z_asr
         α=1 · β=0,10 · γ=0,15 · δ=0,125     ĐÃ CHỐT
```

**Chuẩn hoá z tính μ, σ chỉ trên tập CÓ dữ liệu**, nên `E[z | có dữ liệu] = 0` — đúng
bằng giá trị khung không có dữ liệu nhận được. Có dữ liệu không còn là lợi thế; chỉ
*khớp hơn trung bình* mới là.

RRF thiếu đúng tính chất đó và đo được **−0,2927 FINAL** trên 688 truy vấn.

### `rank` đã bị bác bỏ

| chuẩn hoá | R@10 | R@100 | Final |
|---|---|---|---|
| **z** | **0,460** | **0,708** | **0,4779** |
| rank | 0,035 | 0,150 | 0,0717 |

Thắng 16 / thua 208 · p < 0,0001. **Cơ chế:** `rank` trải đều theo định nghĩa, mà thị
giác phủ 100% khung nên bị dàn đều khắp 173.426 vị trí — hai hạng liền nhau chênh
`6·10⁻⁶`. Một cú khớp OCR khi đó bằng **8,7 lần** toàn bộ khoảng cách hạng 1→1000 của
thị giác (với `z` là 1,2 lần). Bản **trộn** (`z` cho thị giác, `rank` cho nguồn thưa)
cũng thua.

### Trọng số — RÚT TỪ DỮ LIỆU: `1 / 0,0891 / 0,1322 / 0,2128`

Trên **100 truy vấn gán nhãn tay** (`export_for_fusion/`), có nhãn modality thật:
19 vision · 26 vision+ocr · 25 vision+asr · 30 vision+ocr+asr.

| trọng số | R@1 | R@10 | Final | thắng/thua | p |
|---|---|---|---|---|---|
| chỉ thị giác | 0,150 | 0,520 | 0,5200 | — | — |
| **β=γ=δ=0,10** | 0,310 | 0,750 | **0,7420** | 56/3 | **<0,0001** |
| 0 / 0,10 / 0,20 | 0,380 | 0,820 | 0,7760 | 57/1 | <0,0001 |

**Kiểm chéo** — chỉnh trên 50 câu, chấm trên 50 câu giữ kín, 10 lần chia:

    chỉ thị giác              0,5420
    cố định 0,10/0,10/0,10    0,7568
    chỉnh trên nửa A          0,7732     thắng 9/0 so cố định

Gain **tổng quát hoá được**. Nhưng chú ý tỉ lệ: **bật hợp nhất mua +21,5pp**, tinh chỉnh
trọng số chỉ thêm **+1,6pp**. Việc lớn không phải chỉnh số.

Trên bộ 226 câu cũ, cùng phép so cho **p = 0,34** — không phân biệt được với ngẫu nhiên.
Khác biệt **hoàn toàn ở bộ eval**: 226 câu do model *nhìn ảnh* viết ra nên thuần thị giác,
không câu nào cần OCR hay ASR.

**Trọng số tối ưu theo loại** chênh nhau rất xa:

| loại | n | β | γ | δ | Final gốc → tốt nhất |
|---|---|---|---|---|---|
| vision | 19 | 0,10 | 0,10 | 0,10 | 0,621 → 0,684 |
| vision+ocr | 26 | 0 | **0,35** | 0,20 | 0,385 → **0,731** |
| vision+asr | 25 | 0 | 0 | **0,60** | 0,496 → **0,904** |
| vision+ocr+asr | 30 | 0,20 | 0,35 | 0,60 | 0,593 → **0,913** |

⚠️ Dùng được bảng này thì phải **biết loại lúc chạy thi**, mà đề không cho. `query_type`
là nhãn của người annotate. Bộ đoán loại từ nội dung đã đo được **thua** trọng số cố định
— nay có nhãn thật thì kiểm lại được tử tế.

### Bootstrap 200 lần — trọng số và khoảng tin cậy của nó

Lấy mẫu lại 326 truy vấn của hai bộ 200 lần, mỗi lần cực đại `Final` trên lưới 0,0125,
rồi trung bình 200 nghiệm:

| trọng số | trung bình | trung vị | **KTC 5–95%** | xác định? |
|---|---|---|---|---|
| β vật thể | **0,0891** | 0,0875 | [0,049; 0,125] | có |
| γ OCR | **0,1322** | 0,1250 | [0,062; 0,163] | **yếu** |
| δ lời nói | ~~0,1064~~ → **0,2128** | 0,1250 | [0,062; 0,125] | có |

⚠️ **Khoảng tin cậy quan trọng hơn giá trị.** Mỗi trọng số chỉ xác định được trong một
khoảng rộng khoảng **2,5 lần**. `0,0891` không có nghĩa là biết tới chữ số thứ tư — nó
là điểm giữa của một khoảng rất rộng. Đừng chỉnh chữ số thứ ba; hãy mở rộng bộ eval.

### 🔴 δ lời nói ĐÃ ĐỔI 0,1064 → 0,2128 — bộ cũ bị chỉnh THIẾU, không phải khớp quá

Bảng bootstrap trên fit trên **326 truy vấn GỘP hai bộ eval**, mà hai bộ đó khác hẳn nhau:

| bộ | n | % câu cần OCR/ASR |
|---|---|---|
| model viết **khi đang nhìn khung đích** | 226 | **0%** |
| người gán nhãn tay, mô phỏng đề thật | 100 | **81%** |

Gộp lại thì 69% số câu thuộc loại ASR vô dụng, nên nghiệm bị **kéo về 0**. Đây là bài
toán **hỗn hợp**, không phải khớp quá: trọng số fit ra là trung bình của hai chế độ, và
tỉ lệ trộn trong bộ eval không khớp tỉ lệ trong đề thi.

| ×δ | Δ MRR · 100 câu tay | Δ MRR · 226 câu model | **p hoà vốn** |
|---|---|---|---|
| 1,5 | +0,0000 | −0,0234 | 99,9% |
| **2,0** | **+0,0215** | −0,0330 | **60,6%** |
| 2,5 | +0,0438 | −0,0479 | 52,3% |
| 3,0 | +0,0347 | −0,0648 | 65,1% |

`p` = tỉ lệ câu trong đề thi thật cần OCR/ASR. Kiểm chéo 5 lớp × 20 xáo trên bộ 100 câu:
**Δ = +0,0149, KTC95 [+0,0108; +0,0192] ✓**, và **99/100 lớp** chọn ×2,0 hoặc ×2,5 —
tín hiệu, không phải nhiễu chọn lọc.

**Chọn ×2,0.** Giữ được nửa mức lợi và chỉ cần `p > 61%` là có lãi, trong khi bộ gán nhãn
tay đo được `p = 81%`. Không lấy ×2,5 vì nó phụ thuộc nặng hơn vào một `p` chưa biết chắc.

⚠️ Giá trị mới nằm **ngoài** khoảng bootstrap `[0,062; 0,125]` ở bảng trên. Không mâu
thuẫn: khoảng đó là khoảng tin cậy **có điều kiện trên hỗn hợp 326 câu**. Đổi hỗn hợp thì
đổi nghiệm. Muốn thu hẹp thật sự thì phải biết `p` của đề thi.

### Bốn cách rút từ dữ liệu, tất cả rơi vào cùng một vùng

Kiểm chéo lồng 5-fold, chấm `Final` thật trên phần giữ kín:

| cách chọn | Final giữ kín | độ lệch |
|---|---|---|
| lưới tròn `0,10/0,15/0,125` | 0,5546 | 0,055 |
| một lần tìm trực tiếp | 0,5515 | 0,053 |
| **bootstrap trung vị** | 0,5472 | **0,048** |
| bootstrap trung bình | 0,5405 | 0,047 |
| chỉ thị giác | 0,4779 | 0,023 |

Bốn cách chênh nhau **1,4 điểm phần trăm**, nhỏ hơn độ lệch giữa các fold (5 điểm) —
chúng **không phân biệt được**. Điều đáng nói là chúng **đồng thuận về vùng**: bootstrap
độc lập rơi đúng chỗ điểm lưới chọn tay, nên số tròn kia không phải bịa.

**Điều duy nhất thật sự mua được điểm:** có hợp nhất **0,5546** so với chỉ thị giác
**0,4779** — **+7,7 điểm phần trăm**, gấp 5 lần khoảng chênh giữa các cách chọn trọng số.

### Vì sao KHÔNG tinh chỉnh thêm — ba cách khớp bằng toán, cả ba THUA

| cách | Final trên phần giữ kín | so với điểm lưới cố định |
|---|---|---|
| tối ưu **lồi** (softmax NLL) | 0,7020 | −7,4pp |
| tìm **trực tiếp** trên `Final` | 0,7393 | −3,4pp |
| **hàm thay thế trơn của `Final`**, 5-fold, 326 câu | 0,4787 | **−7,6pp · thắng 0/5** |

Cách thứ ba nới lỏng **đúng** công thức, không phải một mục tiêu thay thế:

    hạng(w)  ≈ 1 + Σ_j σ((s_j − s_gt)/τ)
    Final(w) ≈ (1/5) Σ_k σ((k − hạng)/τ₂),   τ hạ dần 2,0 → 0,05

Nó vẫn thua, và trọng số khớp ra **bất ổn**: `(0,63/0,60/1,42)` ở một fold,
`(0/0,06/0,07)` ở fold khác — độ lệch 0,081 so với 0,043 của cách cố định.

**Cơ chế.** `Final` là bậc thang của hạng, hạng là bậc thang của điểm. Gradient chỉ chảy
từ những truy vấn **nằm sát ngưỡng `k`** — số đó ít, và *tập* những câu đó lại đổi theo
`w`. Bộ tối ưu đuổi theo một mẫu nhỏ và trôi. Khi cực trị thật là **cao nguyên rộng** còn
ước lượng thì nhiễu, một điểm cố định giữa cao nguyên đánh bại mọi điểm chọn bằng tối ưu.

⚠️ Điều này **không** mâu thuẫn với việc δ ASR vừa đổi ×2 ở trên. Kết luận ở đây là
"đừng đuổi theo chữ số thứ ba trong CÙNG một hỗn hợp eval"; còn δ đổi vì **hỗn hợp eval
sai**, không vì tinh chỉnh kỹ hơn. Sửa hỗn hợp là việc khác với vặn số.

---

### Top riêng mỗi nguồn: dùng để CHỌN ỨNG VIÊN, không để XẾP HẠNG

Đây là hai câu hỏi khác nhau, và trộn chúng là chỗ tôi từng sai.

**Xếp hạng bằng top riêng — BÁC BỎ.** Lấy top của từng nguồn rồi xen kẽ vòng tròn để ra
thứ tự nộp:

| sinh ứng viên + xếp hạng | Final | Δ |
|---|---|---|
| hợp ĐIỂM rồi lấy top | **0,4473** | — |
| top riêng · vòng tròn 4 nguồn | 0,3043 | **−0,1430** |
| top riêng · bỏ vật thể | 0,3369 | −0,1104 |

Thua nặng vì chia đều 14 mốc cho 4 nguồn nghĩa là **3/4 số ô đầu bảng giao cho nguồn
kém**; riêng vật thể gánh −0,033. Trọng số `0,089 / 0,132 / 0,106` so với `1,0` **chính
là tỉ lệ độ tin cậy đã rút từ dữ liệu**, và xếp hạng vòng tròn vứt bỏ đúng thông tin ấy.

**Chọn ứng viên bằng top riêng — GIỮ**, và đó là kiến trúc hiện tại (`src/retrieval/pool.py`).
Rổ chỉ quyết *ai vào vòng trong*; thứ hạng cuối do ⑤ quyết. Đo với rải 7, chưa có rerank:

| rổ | Final | Δ |
|---|---|---|
| hợp điểm (cũ) | 0,4477 | — |
| **rổ 25/nguồn** | **0,4541** | **+0,0063** |
| rổ 40/nguồn | 0,4489 | +0,0011 |
| rổ 60/nguồn | 0,4477 | −0,0001 |

Trung tính tới hơi lợi — **không còn khoản lỗ 0,1430**. Khác biệt duy nhất so với bản bị
bác bỏ là **ai xếp hạng**. Cơ chế mà rổ mua được: độ phủ video đúng ở ngân sách 16 mốc,
riêng 10 câu nhóm A, tăng từ **10% lên 30%**.

### Đồng thuận giữa các nguồn — tín hiệu MẠNH, dùng làm điểm thì HẠI

`union_pool` trả kèm `provenance`: khung này do những nguồn nào đề cử. Số nguồn đồng thuận
dự báo tính đúng rất mạnh:

| số nguồn đề cử | số khung | là đáp án | tỉ lệ |
|---|---|---|---|
| 1 | 15.119 | 37 | 0,24% |
| 2 | 345 | 34 | **9,86%** |
| 3 | 61 | 17 | **27,87%** |
| 4 | 2 | 1 | 50,00% |
| *nền* | *15.527* | *89* | *0,57%* |

Ba nguồn đồng thuận ⟹ khả năng là đáp án **gấp 48 lần nền**. Nhưng cộng vào điểm:

| hệ số | 0 | 0,25 | 0,5 | 1,0 | 2,0 |
|---|---|---|---|---|---|
| Final | **0,4518** | 0,4311 | 0,4311 | 0,4294 | 0,4294 |

**−0,0208, thắng 3 thua 15** — hại nhất quán. Lý do: chỉ 408/15.527 khung có ≥2 nguồn, và
**72% số khung 3-nguồn vẫn sai**. Thưởng cho chúng đẩy tụt đúng những khung mà thị giác đã
xếp đúng. Tín hiệu có sức phân biệt **khi đứng một mình**, nhưng không thêm gì **khi đã có
điểm** — đúng bẫy lẫn lộn precision với base rate.

Giữ `provenance` lại vì nó là **tín hiệu tin cậy** dùng được cho rải thích ứng (câu chắc
rải dày, câu mơ hồ rải thưa), không phải để làm số hạng trong điểm.

### Hạn ngạch mỗi video — chốt an toàn, không phải phép tối ưu

Một nguồn có thể cho điểm **phẳng trong cả một video**: OCR khớp logo kênh, mà logo hiện ở
mọi khung. Khi ấy top-40 của nó là 40 khung tuỳ tiện của cùng một video. Hạn ngạch
`POOL_PER_VIDEO` chặn việc đó.

| hạn ngạch | không | 20 | **10** | 5 | 3 | 2 |
|---|---|---|---|---|---|---|
| Final | 0,4495 | 0,4495 | **0,4509** | 0,4311 | 0,4300 | 0,4039 |
| nhóm A | 0,0000 | 0,0000 | 0,0000 | 0,0175 | 0,0250 | 0,0250 |

Nới thì **trung tính** (`10` cho Δ=+0,0014, KTC95 [−0,0019, +0,0061] — chứa 0). Siết thì
**hại**: `5` mất 0,0184, `2` mất 0,0456, đổi lại chỉ vớt nhóm A lên 0,0250. Chọn `10` vì
nó chặn được ca bệnh lý với chi phí đo được bằng 0 — **không** vì nó tăng điểm.

---

## ④ DANTE — engine chung của KIS và TRAKE

```
DP[i,t] = S[i,t] + max ( DP[i−1,τ] − λ·(t − τ) )
                  τ < t
```

Tách `t` ra khỏi max cho **max luỹ tiến** ⟹ `O(N·T)` thay vì `O(N·T²)`.

- **Trục thời gian là mili-giây** — `n` bước không đều (p10=19, p90=105 khung);
  `frame_idx` không so được giữa video 25 và 30 fps, mà λ là hằng số chung
- **20/873 video có `pts_time` lùi khi `n` tăng** ⟹ sắp lại lát trước khi vào DP
- `τ < t` **ngặt** ⟹ thứ tự đúng và không hai mốc nào dùng chung một khung

### KIS CHÍNH LÀ TRAKE với N=1 — một đường mã, không hai

`N = 1` làm hệ thức suy biến thành `max_t S[0,t]`. Nên ⑦ chỉ có **một** đường mã cho mọi
loại đề, cài bằng `k_best_alignments` — bản k-best của đúng phép DP trên:

    N=1, k lớn  ⟹ đúng bằng KIS (mọi khung trong rổ, sắp toàn cục theo điểm)
    N>1, k=1    ⟹ đúng bằng TRAKE cũ

⚠️ **Trước đây đây là hai nhánh, và chúng KHÔNG tương đương ở khâu nộp.** Nhánh TRAKE gọi
`dante_over_videos`, vốn trả **một đường mỗi video** — tức khử trùng theo video, mà khử
trùng video đo được **−12,0pp**. Hợp nhất đúng cách là lấy `k` đường mỗi video rồi **sắp
toàn cục**, chứ không phải ép KIS đi qua bản một-đường-mỗi-video.

`tests/test_kis_is_trake_n1.py` khoá tính tương đương lại để hai nhánh không tái sinh:
đường hợp nhất với `N=1` phải cho **đúng** thứ tự của phép sắp thẳng, và
`k_best_alignments(k=1)` phải trùng điểm với `dante()`.

### λ = 0 — mọi λ > 0 đều tệ hơn

| λ | hạng video (trung vị) | mốc đúng chặt | sai số TB |
|---|---|---|---|
| **0** | **6** | **0,50** | 6,6s |
| 0,001 | 3 | 0,67 | 1,9s |
| 0,05 | 206 | 0,11 | 25,6s |
| 1,0 | 131 | 0,06 | — |

**Cơ chế hỏng.** Ở λ=1,0 **cả sáu** truy vấn trả về `0,4s / 1,0s / 1,8s` — ba keyframe
đầu của một video nào đó. `λ(t−τ)` phạt khoảng cách **tuyệt đối** nên nó handicap mọi
đường span lớn ở *mọi* video; trong 873 video luôn có vài video ba khung kề nhau ghi
điểm tàm tạm, thắng chỉ nhờ span ~1 giây. Tức **λ can thiệp vào xếp hạng VIDEO**, trong
khi nó chỉ được thiết kế để tạo hình đường **trong** một video.

λ=0,001 nhìn tốt hơn nhưng **p = 0,69**, và toàn bộ chênh lệch đến từ **một truy vấn**.
Cần ~35 truy vấn TRAKE để kết luận.

---

## ⑤ Rerank — buộc thuộc tính vào đúng vật

**Màu không được buộc vào vật.** Hai câu giống hệt, chỉ đảo màu giữa hai vật:

| | cosine phía chữ | trùng top-20 |
|---|---|---|
| **đảo màu giữa hai vật** | **0,974** | 0,35 |
| chỉ viết lại câu | 0,985 | 0,85 |
| hai câu khác hẳn | 0,42 | 0,00 |

Đảo màu chỉ làm vector chữ đổi **0,026** — gần bằng mức đổi khi chỉ viết lại câu. Tháp
chữ mã hoá `{áo, xe, đỏ, trắng}` như **túi khái niệm**. Đây là lỗi *attribute binding*
của họ CLIP, đo được ngay ở phía chữ. Với **một** vật thì màu rất đúng (áo đỏ 10/10).

**Cắt vật ra rồi mã hoá riêng** buộc màu vào vật bằng cấu trúc. Chấm bằng mắt, 8 truy vấn
màu × 6 khung đầu: **26/48 = 54% → 37/48 = 77%**, không ca nào tệ đi.

Mảnh cắt phủ **một phần**, nên nó là một `SourceScores` đi qua cùng phép chuẩn hoá z của
③ — coi khung không có mảnh là điểm 0 rồi cộng thẳng là đúng lỗi đã giết RRF.

**Trần của tầng này là recall của tầng trước.** `R@10 = 0,407` nhưng `R@1000 = 0,912` ⟹
**50,5 điểm phần trăm nằm ở dải hạng 10–1.000**.

### Đo trên 10 câu nhóm A — trần đó chặt hơn tưởng

*(Đo trên KIẾN TRÚC CŨ: ứng viên là top 100 của điểm đã hợp, chưa có rổ.)* Chạy đúng 10 câu
**sai video** ở phần phân rã bên dưới — nền của chúng là **0,0000 chính xác** (video đúng
vắng mặt), nên mọi thay đổi đều là lãi ròng, phép đo sạch.

| | kết quả |
|---|---|
| kéo được video đúng vào danh sách | **4/10** |
| hạng của chúng | 85 · 71 · 98 · 92 |
| Final trên 10 câu | 0,0163 (nền 0,0000) |
| quy ra toàn bộ 100 câu | **+0,0016** |

**Gần như bằng không, và lý do là hình học của `R@k`.** Điểm lấy `k ∈ {1,5,20,50,100}`,
nên một cú trúng ở hạng 92 chỉ chạm được `R@100` — tức **1/5** điểm — và còn phải rơi
đúng cửa sổ nữa. Rerank kéo video vào được nhưng chỉ tới **đáy bảng**; ở đó điểm gần như
không đổi.

⚠️ Phép đo này **thiên lệch theo thiết kế**: chỉ chạy nơi nền bằng 0 nên chỉ thấy lãi,
không thấy lỗ. ⑤ đảo thứ hạng của cả 81 câu đang trúng. Muốn quyết bật hay tắt mặc định
thì phải chấm đủ 100 câu — xem việc #5 trong *Còn phải làm*.

### Rổ ứng viên đổi hẳn bức tranh — cùng 10 câu, cùng reranker

Chạy lại đúng 10 câu đó bằng kiến trúc rổ + rerank hai bậc:

| | video đúng vào được | hạng của chúng | Final |
|---|---|---|---|
| hợp điểm + ⑤ (cũ) | 4/10 | 71 · 85 · 92 · 98 | 0,0163 |
| **rổ + ⑤b + ⑤c (mới)** | **8/10** | **3 · 7 · 13 · 14 · 21 · 42 · 48 · 48** | **0,0406** |

**Hai tầng làm hai việc khác nhau, và cần cả hai.** Rổ đưa video đúng *vào* danh sách —
đó là thứ hợp điểm không làm được, vì hệ số `1,0` của thị giác dìm mọi bằng chứng thuần
OCR/ASR. Reranker đẩy chúng *lên trên* — hạng 3 và 7 nghĩa là `R@5` và `R@20` bắt đầu ăn
điểm, chứ không chỉ `R@100` như trước.

Đây cũng là lời giải thích vì sao ⑤ đơn độc gần như vô dụng (`+0,0016`): nó chấm lại top
100 của một bảng xếp hạng mà video đúng **không hề có mặt**. Reranker không tạo ra được
ứng viên; nó chỉ sắp lại thứ nó được đưa.

### 🔴 Bậc 1 (mảnh cắt) PHÁ ĐIỂM — đã tắt

Quét đủ 63 tổ hợp trọng số trên bản lưu `_rerank_scores.npz` (không tốn GPU):

| dung hợp 4 nguồn | mảnh cắt | VLM | Final |
|---|---|---|---|
| 1,0 | **0,0** | 0,5 | **0,4700** |
| 1,0 | 0,0 | 1,0 | 0,4593 |
| 1,0 | 0,0 | 0,0 | 0,4513 |
| 0,3 | 1,0 | 1,0 | 0,4184 |
| 0,0 | 0,5 | 0,0 | 0,2331 |
| 0,0 | 2,0 | 0,0 | 0,2331 |

🔴 **CỘT ĐẦU LÀ ĐIỂM DUNG HỢP 4 NGUỒN CỦA ③ — KHÔNG BỎ ĐƯỢC.** Trong mã nó là khoá
`fused4` của `RERANK_WEIGHTS` (trước đây tôi đặt tên `visual`, gây hiểu nhầm là chỉ nguồn
thị giác — đã sửa). Đặt nó về 0 để thứ hạng đến **hoàn toàn** từ reranker thì mất 0,045:

| dung hợp 4 nguồn | mảnh cắt | VLM | Final |
|---|---|---|---|
| **0,0** | 0,0 | 0,5 | **0,4248** ← thứ hạng thuần reranker |
| 0,0 | 1,0 | 1,0 | 0,3943 |
| **1,0** | 0,0 | 0,5 | **0,4700** ← chốt |

Nên kiến trúc đúng là: **rổ lấy top riêng từng nguồn, nhưng thứ hạng vẫn lấy điểm dung
hợp làm xương sống và VLM chỉnh lên xuống.** Reranker bổ sung, không thay thế.

**Mọi tổ hợp có `mảnh cắt ≥ 0,5` đều tụt, càng nặng càng tệ.** Nên `crop = 0,0`, và mã
hiểu trọng số 0 là **không tính luôn** chứ không tính rồi bỏ — tiết kiệm **$0,76 và ~8
phút** mỗi lượt chạy 100 câu, đúng thứ cần cho kỳ thi 2 giờ 30.

⚠️ **Không mâu thuẫn phép đo "màu đúng 54% → 77%"** ở đầu mục này: phép đo đó chấm bằng
mắt trên **8 truy vấn thuần màu**, còn bộ 100 câu gần như không có câu nào phụ thuộc màu.
Mảnh cắt nhiều khả năng vẫn lợi ở loại đề đó — bật lại bằng cách đặt `crop > 0` trong
`RERANK_WEIGHTS`. Cần một bộ eval màu đủ lớn mới kết luận dứt điểm được.

### Lấy đỉnh là KHỚP QUÁ — và đây là lượng khớp quá, đo được

Quét 63 tổ hợp rồi lấy đỉnh trên **chính bộ dùng để đo** là thiên lệch chọn lọc. Phép
kiểm đúng là kiểm chéo lồng nhau: chọn trọng số trên 80 câu, chấm trên 20 câu mà **quá
trình chọn chưa từng thấy**. 5 lớp × 20 lần xáo:

| | Δ so nền cũ |
|---|---|
| đỉnh trên bộ đầy đủ `(2,0 / 0,0 / 0,25)` | +0,0241 ← **thiên lệch** |
| **kiểm chéo** | **+0,0175** · KTC95 [+0,0147, +0,0201] ✓ |
| **lượng KHỚP QUÁ** | **+0,0066** (27% mức lợi biểu kiến) |

**Lợi ích là thật nhưng nhỏ hơn con số quét ra.** Số nên tin là **+0,0175**.

Hai điều kiểm chéo nói thêm:

- **`mảnh cắt = 0` rất vững** — 88/100 lần chia lớp đều chọn tổ hợp có `crop = 0,0`. Kết
  luận tắt bậc 1 không phải nhiễu chọn lọc.
- **Vùng phẳng rộng** — 0,4767 xuống 0,4739 trải khắp `dung hợp ∈ [0,5; 2,0]` và
  `VLM ∈ [0,25; 1,0]`, tức tỉ lệ **2:1 tới 8:1 đều như nhau**. Chỉ tỉ lệ có nghĩa, nên
  chuẩn hoá `dung hợp = 1` rồi lấy `VLM = 0,25` (giá trị được chọn ở 63/100 lớp) —
  **điểm ổn định giữa vùng phẳng, không phải đỉnh**. Cùng lập luận bias–variance đã dùng
  khi chốt trọng số 4 nguồn.

---

## ⑥ Đầu đọc — chỉ cho QA

Qwen2.5-VL-7B đọc khung top-1 + **OCR + lời nói** của khung đó → sinh `answer`. Đưa kèm
chứng cứ chữ vì nhiều câu hỏi (tên riêng, con số, ngày tháng) **chỉ đọc được từ chữ**.

Không có tầng này thì câu QA **được 0 điểm** dù tìm đúng khung: thể lệ đòi `aᵢ = GTₐ`.

⚠️ **Chưa có bộ eval QA nào.** Ở ca thử duy nhất, truy xuất tìm đúng cảnh nhưng đầu đọc
trả lời **"Ba"** trong khi khung có **2 người** — sai.

---

## ⑦ Nộp bài

Ngân sách **100 câu trả lời** mỗi truy vấn. `R@k = max_{i≤k} R(rᵢ)` với
`k ∈ {1,5,20,50,100}`, `Final = (1/5)·Σ R@k` — nên vị trí trong danh sách có trọng số
rất lệch: câu số 1 được tính vào cả 5 mức, câu số 51–100 chỉ 1 mức.

### KHÔNG khử trùng — một kết luận đã bị ĐẢO

Khử trùng theo cảnh từng đo được **+2,0pp** (thắng 23/thua 0, p<0,0001). Phép đo đó giả
định `đáp án = keyframe của ta` — điều thể lệ bác bỏ. Đo lại theo mô hình đúng:

| khử trùng | rải | L=9 | L=11 | L=21 |
|---|---|---|---|---|
| **không** | **7** | **0,2317** | **0,2476** | 0,2788 |
| cảnh | 7 | 0,1889 | 0,2037 | 0,2408 |
| không | 5 | 0,2120 | 0,2406 | **0,2974** |
| không | 1 | 0,0831 | 0,1022 | 0,1844 |

Khử trùng cảnh **mất 4,3pp** (−23% tương đối). Theo video còn tệ hơn.

**Cơ chế:** phép rải **đã tự lo việc chống trùng**. Mỗi keyframe chỉ rải trong *nửa khe
của chính nó*, nên các dải rải **lát kề nhau, không chồng lên nhau**. Khử trùng sau đó
chỉ bỏ phần **phủ** mà không bỏ được phần **trùng** nào — vì không còn phần trùng nào.

### Đáp án KHÔNG phải keyframe của ta — trần cứng 23,5%

Thể lệ nói rõ ba điều:

> *"**khung hình ngữ nghĩa** … **khác với** I-Frame là khung hình kỹ thuật … **đã được
> cung cấp cho các đội thi**"*
>
> *"đoạn ứng với khoảnh khắc ngữ nghĩa này **thường rất ngắn, thông thường là dưới 10
> frame**"* — và *"cùng nguyên tắc … như ở Textual KIS và Q&A"*
>
> Ví dụ KIS: đáp án `[500, 510]` = **11 khung**

Keyframe của ta cách nhau **trung vị 48 khung** (p10=19, p90=105). Cửa sổ 11 khung lọt
gọn vào khe. Xác suất một cửa sổ rộng `L` đặt bất kỳ **chứa sẵn** keyframe:

    L=9 → 23,5%    L=11 → 27,9%    L=25 → 57,6%    L=51 → 85,1%    L=101 → 96,6%

**Ở `L ≈ 10`, nộp thuần keyframe có trần cứng ~23%** — ba phần tư số câu thua vì **hình
học**, không phải vì ngữ nghĩa.

**Sửa: rải khung vào khe.** Bảng trên cho `0,0831 → 0,2317` ở `L=9` — **×2,8**.

**Hệ quả: 96,3% ô nộp là khung ta CHƯA BAO GIỜ cắt.** Trong 10.000 dòng của bài nộp thật,
chỉ **373** trùng một keyframe có trong chỉ mục; 9.627 dòng còn lại là số khung ta không
có ảnh, không có vector, không có OCR. Đó là **chủ ý**, không phải lỗi: bài nộp ghi
`frame_idx` trên **video gốc**, không phải trên tập keyframe của ta, và thể lệ nói rõ khung
ngữ nghĩa khác I-Frame được cấp.

Hai chặn an toàn cho các khung "ảo" này, cả hai đang bật: rải bị cắt theo **biên shot**
(`shot_start_frame`/`shot_end_frame`) nên không tràn sang cảnh khác, và theo **số khung
thật của video** nên không sinh `frame_id` vượt độ dài — đúng lỗi từng bắt được.

Điều này **không làm suy yếu ⑤**: reranker chạy trên **mốc**, tức keyframe ta có ảnh, và
chạy **trước** khi rải. Rải là hậu xử lý thuần, nên 9.627 khung ảo không bao giờ đi vào
reranker. Đổi lại, ⑤ cũng **không thể** sửa nhóm C — nó chỉ chọn được keyframe nào, không
biết mốc ngữ nghĩa nằm đâu trong khe.

Cái giá của rải: 7 khung/mốc nghĩa là 100 ô chỉ mua được ~14 mốc, phủ **trung vị 6 video**
(min 1, max 15). Đó chính là nguồn gốc nhóm A ở phần phân rã bên dưới.

🔴 **MẶC ĐỊNH HIỆN TẠI LÀ `spread = 1` (KHÔNG RẢI), THEO YÊU CẦU — và nó mất 3,3 lần
điểm.** Đo trên cùng 100 truy vấn, mô hình mốc lệch:

| | rải 7 | **rải 1** |
|---|---|---|
| hợp điểm (cũ) | 0,4477 | **0,1363** |
| rổ 40/nguồn (mới) | 0,4489 | **0,1366** |

Đây **không phải** vấn đề xếp hạng nên reranker không cứu được: trần hình học của bài nộp
thuần keyframe là **~23,5%**, mà bản không rải đã ở 0,1366 — tức đã dùng 58% của trần đó.
Bản có rải đang ở 0,4477, **cao gấp đôi cái trần ấy**. Bật lại bằng `--spread 7`.

Phần dưới là kết quả chỉnh `spread` khi nó còn bật, giữ lại vì `--spread 7` vẫn dùng được.

**Chọn `spread = 7` vì nó nằm giữa một VÙNG PHẲNG, không vì nó thắng ai.**

| spread | 3 | 5 | **6** | **7** | **8** | **9** | 11 | 14 |
|---|---|---|---|---|---|---|---|---|
| số mốc | 33 | 20 | 16 | 14 | 12 | 11 | 9 | 7 |
| Final | 0,2534 | 0,4136 | **0,4488** | **0,4467** | **0,4485** | **0,4464** | 0,4366 | 0,4096 |

Sáu tới chín **không phân biệt được**: đỉnh danh nghĩa `spread=6` hơn 7 đúng **+0,0021,
KTC95 [−0,0092, +0,0136]**, thắng 38 thua 27 — nhiễu thuần. Vùng phẳng có biên thật ở 5
và 11; ngoài đó tụt rõ. Giữ 7 vì nó là đương nhiệm và nằm giữa vùng phẳng.

⚠️ **Lý lẽ cũ ở đây SAI và đã bị thay.** Trước đây tôi phá hoà giữa 7 và 9 bằng cách chấm
theo hai mô hình `chặt ±4` và `cửa sổ khe`, rồi kết luận "bảy thắng cả hai". Nhưng cả hai
mô hình đó **ôm keyframe ground truth**, mà `spread=7` phát ra 14 keyframe còn `spread=9`
chỉ phát 11 — nên phép so ấy tự thưởng cho phương án phát nhiều keyframe hơn. Bảng trên đo
lại bằng **mô hình mốc lệch**, và nó xoá luôn khoảng cách: 0,4467 so với 0,4464. Kết luận
giữ nguyên, **lý do thì không**.

⚠️ Tôi cũng từng chốt 9 bằng lập luận Định lý 1 — khe trung vị 48 chia 8 cho bước 6, bảo
đảm khi `L ≥ 6`, so với bước 8 của `m=7`. Lập luận đó đúng về bảo đảm nhưng sai về đánh
đổi: nó bỏ qua chi phí mất 3 mốc. Cả hai lần, **định lý và mô hình thiên lệch đều thua
phép đo không thiên lệch**.

Lưới **thích ứng theo khe cục bộ**, không bước cố định — khe chênh hơn 5 lần giữa p10 và
p90. Đây là hướng còn mở: vùng phẳng 6–9 nói rằng *bước cố định nào cũng thế*, nhưng bước
**tỉ lệ với khe của chính mốc đó** chưa được đo.

⚠️ **Ngưỡng đảo chiều ≈ `L = 60`.** Nếu cửa sổ thật rộng hơn thế thì `--spread 1` mới
đúng.

---

## Tiền còn nằm ở đâu — phân rã 19 câu trượt

Chấm bài nộp thật của 100 truy vấn gán nhãn tay: **19 câu trượt cả hai mô hình cửa sổ**.
"Trượt" gộp **ba lỗi khác hẳn nhau**, và cách chữa của chúng ngược nhau — nên phải tách
trước khi sửa bất cứ thứ gì.

| nhóm | n | triệu chứng | tầng chịu trách nhiệm |
|---|---|---|---|
| **A · sai video** | 10 | video đúng vắng mặt hoàn toàn trong 100 ô | ③ xếp hạng, ⑤ rerank |
| **B · sai khoảnh khắc** | 6 | đúng video, lệch **trung vị 515 khung** | ④ DANTE |
| **C · hụt phủ** | 3 | đúng mốc, lệch 13–90 khung | ⑦ rải |

**Nhóm A không phải lỗi ngữ nghĩa mà là lỗi ngân sách.** 9/10 câu có video đúng nằm ở
**hạng mốc 16–62** — hoàn toàn với tới được; chỉ một câu ở hạng 817 là vô vọng. Ta cắt ở
mốc thứ 14 vì rải 7 khung ăn hết 100 ô, nên bài nộp chỉ phủ **trung vị 6 video** (min 1,
max 15).

**Nhóm B có một ca chẩn đoán được.** Câu `vision+ocr` "chiến sĩ bộ đội múa rồng trên kênh
HTV" xếp video **hạng 1** nhưng lệch **5.068 khung**. OCR khớp logo kênh vốn hiện suốt cả
video, nên nguồn này cấp tín hiệu **cấp video chứ không phải cấp khoảnh khắc**. Đó là
hành vi hỏng có địa chỉ, không phải nhiễu — một chặn phương sai theo video cho OCR sẽ
tách được hai vai trò đó.

### Chia lại ngân sách KHÔNG cứu được nhóm A

Giả thuyết tự nhiên: bớt khung mỗi mốc để phủ nhiều mốc hơn. Quét `k` mốc đầu rải 7 khung,
phần còn lại rải 1 khung, tổng luôn đúng 100 ô, chấm theo **mô hình mốc lệch ngẫu nhiên**:

| k | 0 | 3 | 5 | 6 | 12 | 16 (≈ đang dùng) |
|---|---|---|---|---|---|---|
| Final | 0,1424 | 0,4288 | 0,4488 | 0,4493 | **0,4516** | 0,4507 |

**Bằng phẳng từ k=5 tới k=16.** Đỉnh `k=12` hơn cấu hình hiện tại **+0,0009, KTC95
[−0,0008, +0,0029]** — chứa 0. Mỗi ô lấy khỏi độ phủ khe mua về đúng bằng ngần ấy độ phủ
video; hai áp lực triệt tiêu nhau. Cùng dạng bias–variance như lúc chốt trọng số: **giữ
điểm cố định trong vùng phẳng, không đuổi theo argmax.**

Kết luận: nhóm A phải chữa bằng cách làm video đúng **leo lên** top 14, không phải bằng
cách đào sâu hơn.

---

## Ý tưởng đã BÁC BỎ bằng đo

Giữ lại để không ai dựng lại chúng.

| ý tưởng | kết quả |
|---|---|
| RRF | −0,2927 FINAL trên 688 truy vấn |
| Đa biến thể probe (max/mean) | xấu hơn 17/30 câu |
| Rút gọn truy vấn (cắt câu dẫn) | lợi 1,29× không bù rủi ro cắt nhầm 2,4× |
| Chuẩn hoá `rank`, và bản trộn | 0,035 vs 0,460 |
| Khử trùng theo **video** | −12,0pp |
| Token 2 âm tiết cho BM25 tiếng Việt | trung tính tới hơi tệ trên cả ba nguồn |
| Câu bối cảnh trước `E1:` để chọn video | hạng video trung vị 6 → 51 |
| Định tuyến trọng số theo loại truy vấn | thua trọng số cố định trên cả hai trục |
| Tách câu + `MIN` để buộc màu | tách tốt hơn nhưng khung trả về sai |
| Mã hoá thêm hai mảnh bên (chống cắt giữa) | bước nhảy tại biên chỉ 1,07× |
| **Xếp hạng** bằng top riêng mỗi nguồn (xen kẽ vòng tròn) | −0,1430 Final; vứt mất tỉ lệ tin cậy đã đo. *Dùng top riêng để **chọn ứng viên** thì khác — cái đó GIỮ, xem ③* |
| Siết hạn ngạch mỗi video xuống 2–5 | −0,018 tới −0,046; chỉ vớt nhóm A lên 0,025 |
| Chia lại ngân sách nộp phi đồng đều (k mốc dày + phần còn lại thưa) | vùng phẳng k=5…16; đỉnh hơn +0,0009, KTC chứa 0 |

---

## Còn phải làm

Sắp theo **giá trị kỳ vọng**, việc đã xong gom xuống cuối.

| # | việc | vì sao | chi phí |
|---|---|---|---|
| 1 | 🔴 **Quyết lại `spread`** | mặc định nay là `1` theo yêu cầu, và nó **mất 3,3 lần điểm** (0,4477 → 0,1366). Trần hình học 23,5% chặn cứng — reranker không cứu được. Bật lại: `--spread 7` | $0 |
| 2 | **Biết `p`** — tỉ lệ đề thi thật cần OCR/ASR | quyết định trực tiếp δ ASR: `p>61%` thì ×2 có lãi, `p>52%` thì ×2,5 có lãi. Đây là **tham số quan trọng nhất còn chưa biết** | $0 |
| 2b | 🔴 **Thu hẹp khoảng cách `R@1 ↔ R@100`** | dư địa **+0,2422** — gấp 7 lần tổng mọi khoản tinh chỉnh trọng số. Rổ đã đưa đáp án vào top 100 cho 73% câu, nhưng chỉ 11% được đặt hạng 1 | $0 |
| 3 | 🔴 **Chữa `vision+ocr`** | loại **thấp điểm nhất** theo thước BTC: 0,4846 ở `L=11`, thấp hơn cả `vision` thuần (0,5158). Đây là bốn TẬP truy vấn khác nhau nên không kết luận được "thêm OCR thì tệ đi" — nhưng nó nói câu cần OCR đang **khó nhất** với hệ, dù OCR là nguồn ta có dữ liệu đầy đủ. Đông thứ hai (26/100) | $0 |
| 3b | **Bộ eval MÀU đủ lớn** | mảnh cắt lợi 54%→77% trên 8 câu thuần màu nhưng hại −0,03 trên bộ 100 câu ít màu. Không mâu thuẫn — chỉ là chưa biết tỉ lệ đề màu thật | $0 |
| 4 | **Bộ eval QA** | ⑥ chưa có phép đo nào; ca thử duy nhất trả lời sai | $0 |
| 5 | **Chốt λ** | hiện 6 truy vấn TRAKE, p=0,69 — cần ~35 | $0 |
| 6 | Quét `k1`, `b` của BM25 | chưa xong: chạy cục bộ quá chậm (0/20 tổ hợp sau nhiều phút), phải đưa lên Modal | ~$0,10 |
| 7 | **Rải theo khe CỤC BỘ** thay bước cố định | vùng phẳng 6–9 nói bước cố định nào cũng thế; bước tỉ lệ khe của chính mốc chưa đo. Nhóm C hỏng đúng vì khe **bất đối xứng** | $0 |
| 8 | Dựng bộ **đoán loại truy vấn** | bảng theo loại chênh tới +41pp nhưng cần biết loại; nay đã có 100 nhãn thật để huấn luyện | $0 |
| 9 | `modal.Cls` + `@modal.enter()` cho 3 hàm GPU | hiện `from_pretrained` nằm TRONG thân hàm ⟹ mỗi container nạp lại mô hình. Với `.map()` thì trả phí đó nhiều lần | $0 |
| 10 | `writer.py` validator | chờ **mẫu file nộp chính thức** của BTC | $0 |

<details><summary><b>Đã xong trong vòng này</b></summary>

| việc | kết quả |
|---|---|
| Rổ ứng viên hợp top riêng mỗi nguồn | nhóm A: 4/10 → **8/10** video, hạng 71–98 → **3–48** |
| Rerank 2 bậc + quét 63 tổ hợp trọng số | **tắt bậc 1** (mảnh cắt phá điểm), giữ bậc 2 |
| Kiểm chéo trọng số rerank | +0,0175 sau kiểm chéo, khớp quá +0,0066 |
| Kiểm chéo trọng số dung hợp | δ ASR ×2 — **+0,0149** sau kiểm chéo |
| Hợp nhất ⑦: KIS = TRAKE N=1 | một đường mã; nhánh cũ khử trùng video ngầm (−12,0pp) |
| `.remote()` tuần tự → `.map()` | mã hoá **31× nhanh hơn**; 100 câu từ ~3 giờ xuống ~5 phút |
| Lưu điểm ⑤ trung gian | mọi phép quét trọng số rerank về sau là **$0** |
| Hạn ngạch mỗi video | **kết quả âm** — nới thì trung tính, siết thì hại. Giữ 10 làm chốt an toàn |
| Đồng thuận giữa các nguồn | **kết quả âm** — tín hiệu mạnh (gấp 48× nền) nhưng cộng vào điểm thì hại |
| Đo lại theo mô hình đúng | đảo một kết luận: khử trùng cảnh từ +2,0pp thành **−4,3pp** |

</details>

### Ba hướng tối ưu không cần khung mới

1. ~~**Số mốc × số khung thích ứng theo độ tin cậy**~~ — **đã đo, vùng phẳng.** Quét phân
   bổ phi đồng đều (k mốc dày + phần còn lại thưa) cho k=5…16 đều ngang nhau; đỉnh hơn
   cấu hình hiện tại +0,0009 với KTC chứa 0. Mỗi ô lấy khỏi độ phủ khe mua về đúng ngần
   ấy độ phủ video. Vẫn còn cửa nếu **điều kiện hoá theo độ tin cậy từng câu** thay vì
   theo hạng — nhưng chưa có tín hiệu tin cậy nào đã hiệu chuẩn.
2. **Rải theo biên CẢNH thay vì nửa khe keyframe** — `shot_id` cho biết ranh giới cảnh
   thật; nếu mốc ngữ nghĩa chắc nằm trong cảnh thì rải trong biên cảnh chính xác hơn.
3. **Đo lại `spread` sau khi có ⑤** — rerank đổi thứ hạng nên số mốc đáng giữ có thể khác.

---

## Bài học đo lường

**Kiểm định phải chạy trên `Final`, không phải trên hạng thô.** Hạng đổi ở vùng sâu
(200 → 400) **không đổi `R@k` nào** nên không đổi điểm. Đo trên hạng thô cho ⑤ bậc 2 ra
"36/2"; đo trên `Final` ra "11/1" — cùng dữ liệu, hai bức tranh khác hẳn, và chỉ bức
thứ hai nói đúng thứ cuộc thi chấm.

**Mô hình cửa sổ ÔM keyframe ground truth là phép đo khép kín — nó thưởng cho việc nộp
lại chính cái nhãn của mình.** Cả `chặt ±4` lẫn `cửa sổ khe` đều dựng quanh keyframe GT,
nên nộp đúng keyframe đó là ăn điểm chắc chắn. Hệ quả: mọi so sánh làm **thay đổi số
keyframe được phát ra** đều bị thiên lệch. Quét phân bổ ngân sách theo hai mô hình này cho
"không rải" đứng đầu với `0,7520`; đo lại theo mô hình **mốc lệch ngẫu nhiên** thì đúng
cấu hình ấy tụt xuống `0,1424` — **kết luận đảo ngược hoàn toàn**.

Hai dấu hiệu đã tố cáo lỗi trước khi tôi kịp tin vào nó: (1) `0,7520` **giống hệt nhau ở
cả hai cột**, mà hai mô hình khác nhau thì không thể cho cùng một số; (2) nó mâu thuẫn
với trần **23,5%** đã đo độc lập. Quy tắc rút ra: **chỉ dùng mô hình mốc lệch để so các
phương án rải/phân bổ.** Hai mô hình kia chỉ dùng để chặn trên và chặn dưới điểm cuối.

**Con số tự mâu thuẫn là dấu hiệu lỗi phép đo, không phải phát hiện.** Lần đầu chạy ⑤
bậc 2, mọi `R@k` đều tăng nhưng kiểm định ra "thua 36/60". Hai điều đó không thể cùng
đúng — hoá ra nhãn thắng/thua bị đảo do `zip(B, A)` đặt tên biến ngược. Nếu chỉ nhìn
p-value mà không đối chiếu bảng `R@k` thì đã báo cáo ngược hoàn toàn.

## Thiên lệch phải biết

Truy vấn trong `data/eval/` do người hoặc model **nhìn khung đích** viết ra, nên câu chữ
có thể vô tình khớp cách encoder biểu diễn. Đo được câu tôi viết tay ăn điểm **~1,5×**
câu Qwen viết. Mọi con số trên trang này là **cận trên**, và cần một phần truy vấn do
người khác viết để kiểm chéo.

---

## Bố cục

```
scripts/run.py                 đường chạy tổng — ①②③④⑤⑥⑦
scripts/index/                 dựng lại chỉ mục, chạy theo số thứ tự
  1_probe_model.py             soi hành vi encoder TRƯỚC khi tiêu tiền
  2_encode_frames.py           mã hoá 173.426 khung (~$4,32)
  3_build_index.py             gộp .npy theo video → chỉ mục phẳng ($0)
  4_write_manifest.py          provenance: SHA model, preprocessing, hash
scripts/eval/                  công cụ đo, không thuộc đường nộp bài
  encode_queries.py            mã hoá truy vấn eval, cache lại
  score_index.py               ba phép đo chỉ mục, không cần nhãn tay
  score_submission.py          chấm ĐÚNG LUẬT BTC, quét bề rộng `L`
  compare_arch.py              so kiến trúc; suy `spread` khác từ bản lưu ⑤
  build_bench_artifact.py      dựng `bench100.html` — soi từng câu kèm ảnh

src/  dựng chỉ mục
  ingestion/jina_encoder.py    Matryoshka: CẮT rồi mới chuẩn hoá
  ingestion/vector_index.py    chỉ mục phẳng, mang cả `n` lẫn `frame_idx`
  index/shard_plan.py          chia lô giữa container + chiếu chi phí
src/  truy vấn
  retrieval/probe.py           ①  tách mốc, rút trích dẫn
  retrieval/sources.py         ②  bốn nguồn + `covered`
  retrieval/bm25.py                BM25, cố tình KHÔNG tự bỏ dấu
  retrieval/score_matrix.py    ③  chuẩn hoá z trong tập có dữ liệu
  retrieval/pool.py            ⑤a rổ ứng viên: hợp top RIÊNG mỗi nguồn
  retrieval/dante.py           ④  DP O(N·T), trục mili-giây
  retrieval/rerank.py          ⑤bc mảnh cắt (bậc 1) + VLM (bậc 2)
  submission/kbest.py          ④⑦ k-best của cùng DP — KIS là N=1
  submission/coverage.py       ⑦  Định lý 1 + rải khung vào khe
src/text/fold.py               bỏ dấu tiếng Việt — MỘT cài đặt duy nhất

artifacts/embed/               provenance manifest
data/eval/                     ground truth (226 KIS · 6 TRAKE · 30 màu)
queries/                       thả file .txt vào đây
submission/                    đầu ra
```
