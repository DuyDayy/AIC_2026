# Tầng truy vấn — AIC 2026

Hệ truy xuất video đa phương thức cho vòng sơ tuyển. Tiền xử lý (khung hình · ASR · OCR ·
vật thể) đã xong; kho này là **tầng truy vấn**: từ đề bài ra bài nộp.

```
⓪ mã hoá offline  173.426 khung → lưu 1024 chiều, tra ở 512
      │
đề → ① probe → ② bốn nguồn → ③ ma trận S → ④ DANTE ─┬──→ ⑤ rerank → ⑦ nộp
                                                     └─ QA ─→ ⑥ đầu đọc ─┘
```

---

## Chạy

Thả file `.txt` vào `queries/`, mỗi truy vấn một file. Tên file thành `query_id`.

```bash
modal run scripts/run.py                    # ./queries → ./submission
modal run scripts/run.py --no-rerank        # bỏ ⑤
modal run scripts/run.py --light            # chỉ thị giác, cho máy thiếu RAM
modal run scripts/run.py --spread 1         # nộp thuần keyframe (xem ⑦)
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
         α=1  ·  β=γ=δ=0,10  (TẠM)
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

### Trọng số — TẠM, chờ nhãn tay

| trọng số | R@10 (tả cảnh) | kiểm định dấu | ca sự kiện |
|---|---|---|---|
| β=γ=δ=0 | 0,407 | — | hạng 67 |
| β=0,10 một mình | 0,456 | 98/72 · p=0,055 | hạng 89 |
| γ=0,10 một mình | 0,425 | 51/97 · **p=0,0002 → HẠI** | hạng 3 |
| **β=γ=δ=0,10** | **0,460** | 99/85 · p=0,34 | **hạng 1** |

γ (OCR) hại câu tả cảnh nhưng cứu câu sự kiện. Không nghịch lý: 226 câu tả cảnh do model
**nhìn ảnh** viết ra nên thuần thị giác. Đo trên 30 đề thi thật thì **40% nhắc chữ trên
khung**, 20% có tên riêng ⟹ bộ eval hiện tại thiên lệch **chống lại** OCR/ASR.

0,10 là mức thấp có chủ ý: đủ cứu câu sự kiện, chưa đủ gây hại đo được.

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

### Khử trùng theo CẢNH — và VIDEO là bẫy

| cách nộp | Final |
|---|---|
| thô | 0,5168 |
| **khử trùng cảnh** | **0,5372** (+2,0pp · thắng 23/thua 0 · p<0,0001) |
| khử trùng video | 0,3965 (**−12,0pp**) |

Khử trùng video hại nặng vì khi đã đúng video thì **nhiều khung trong đó cùng nằm trong
cửa sổ đáp án** — ép mỗi video một khung là tự vứt các cơ hội ấy.

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

**Sửa: rải khung vào khe.** 226 truy vấn, mốc thật lệch ngẫu nhiên trong khe:

| cách nộp | L=9 | L=11 | L=15 | L=21 |
|---|---|---|---|---|
| 100 mốc × 1 khung | 0,0608 | 0,0758 | 0,1002 | 0,1384 |
| 20 mốc × 5 khung | 0,1624 | 0,1877 | **0,2137** | **0,2349** |
| **14 mốc × 7 khung** ← mặc định | **0,1742** | **0,1882** | 0,2038 | 0,2184 |

**×2,5.** Chọn `spread=7` vì hai lý do cộng lại: tốt nhất ở `L=9` — điểm vận hành nhiều
khả năng nhất — và 7 khung trong khe trung vị 48 cho **bước 8 khung**, nên theo Định lý 1
nó **bảo đảm** trúng khi `L ≥ 8`, không còn là xác suất.

Lưới **thích ứng theo khe cục bộ**, không bước cố định — khe chênh hơn 5 lần giữa p10 và
p90.

⚠️ **Ngưỡng đảo chiều ≈ `L = 60`.** Nếu cửa sổ thật rộng hơn thế thì `--spread 1` mới
đúng.

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

---

## Còn phải làm

| # | việc | vì sao | chi phí |
|---|---|---|---|
| 1 | **Đo lại toàn bộ theo mô hình đúng** | mọi Final trước đây dùng giả định *đáp án = keyframe*, mà thể lệ đã bác bỏ ⟹ vài kết luận có thể đảo | $0 |
| 2 | **Bộ eval QA** | ⑥ chưa có phép đo nào; ca thử duy nhất trả lời sai | $0 |
| 3 | **Chốt `β, γ, δ`** | cần truy vấn **gán nhãn tay**, ưu tiên câu có tên riêng / chữ trên màn hình / lời thoại | $0 |
| 4 | **Chốt λ** | hiện 6 truy vấn TRAKE, p=0,69 — cần ~35 | $0 |
| 5 | Rerank mảnh cắt quy mô đầy đủ | 54%→77% mới đo trên 8 câu | ~$0,15 |
| 5b | Quét `VLM_WEIGHT` rồi quyết có bật ⑤ bậc 2 mặc định không | xác suất VLM đã tính sẵn, chỉ đổi trọng số | $0 |
| 6 | Quét `k1`, `b` của BM25 | chưa đụng; phải chạy trên Modal vì máy cục bộ hết RAM | ~$0,10 |
| 7 | Nối `allocator` / `kbest` vào ⑦ | `kbest` cho TRAKE k-best; hiện chỉ lấy đường DP tốt nhất mỗi video | $0 |
| 8 | `writer.py` validator | chờ **mẫu file nộp chính thức** của BTC | $0 |

### Ba hướng tối ưu không cần khung mới

1. **Số mốc × số khung thích ứng theo độ tin cậy** — câu chắc thì ít mốc rải dày, câu mơ
   hồ thì nhiều mốc rải thưa. Hiện cố định 14×7 cho mọi câu.
2. **Rải theo biên CẢNH thay vì nửa khe keyframe** — `shot_id` cho biết ranh giới cảnh
   thật; nếu mốc ngữ nghĩa chắc nằm trong cảnh thì rải trong biên cảnh chính xác hơn.
3. **Đo lại `spread` sau khi có ⑤** — rerank đổi thứ hạng nên số mốc đáng giữ có thể khác.

---

## Bài học đo lường

**Kiểm định phải chạy trên `Final`, không phải trên hạng thô.** Hạng đổi ở vùng sâu
(200 → 400) **không đổi `R@k` nào** nên không đổi điểm. Đo trên hạng thô cho ⑤ bậc 2 ra
"36/2"; đo trên `Final` ra "11/1" — cùng dữ liệu, hai bức tranh khác hẳn, và chỉ bức
thứ hai nói đúng thứ cuộc thi chấm.

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

src/  dựng chỉ mục
  ingestion/jina_encoder.py    Matryoshka: CẮT rồi mới chuẩn hoá
  ingestion/vector_index.py    chỉ mục phẳng, mang cả `n` lẫn `frame_idx`
  index/shard_plan.py          chia lô giữa container + chiếu chi phí
src/  truy vấn
  retrieval/probe.py           ①  tách mốc, rút trích dẫn
  retrieval/sources.py         ②  bốn nguồn + `covered`
  retrieval/bm25.py                BM25, cố tình KHÔNG tự bỏ dấu
  retrieval/score_matrix.py    ③  chuẩn hoá z trong tập có dữ liệu
  retrieval/dante.py           ④  DP O(N·T), trục mili-giây
  retrieval/rerank.py          ⑤  mảnh cắt buộc thuộc tính vào vật
  submission/coverage.py       ⑦  Định lý 1 + rải khung vào khe
src/text/fold.py               bỏ dấu tiếng Việt — MỘT cài đặt duy nhất

artifacts/embed/               provenance manifest
data/eval/                     ground truth (226 KIS · 6 TRAKE · 30 màu)
queries/                       thả file .txt vào đây
submission/                    đầu ra
```
