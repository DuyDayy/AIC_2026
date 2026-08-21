# Tầng truy vấn — AIC 2026

Hệ truy xuất video đa phương thức, vòng sơ tuyển. Tiền xử lý (khung hình · ASR · OCR) đã
xong; kho này là **tầng truy vấn**: từ đề bài ra bài nộp.

Vận hành ngày thi: [`RUNBOOK.md`](RUNBOOK.md) · Tối ưu: [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md)

```
⓪ MỘT LẦN, offline          609.476 khung · 873 video · lưu 1024 chiều, tra ở 512


A ─ CHUẨN BỊ THEO LÔ ─ scripts/rf_prepare.py ─ chạy TRƯỚC khi nhận đề

   gói đề .txt                     ① probe + mở rộng
   ├─ *-kis  → 1 phiên             ├─ tách mốc E1:/E2:
   ├─ *-qa   → 1 phiên             ├─ Q&A qua declarativize
   └─ *-trake→ N phiên             └─ đọc expansions/
      (mỗi event thành MỘT câu KIS)         │
                                            ▼
                            ② 7 run          ③ DUNG HỢP RRF hai tầng
                            1 thị giác       beta  expansion trong modality
                            3 OCR            alpha modality với nhau
                            3 ASR                   │
                                                    ▼
                              data/rf_sessions/<id>/
                                meta.json     đề · loại · event_index
                                vectors.npz   q₀ + text_rrf ĐỦ 609.476 ← ghi 1 lần
                                events.jsonl  nhật ký thao tác     ← nối thêm


B ─ TƯƠNG TÁC ─ scripts/rf_server.py ─ operator ngồi trước màn hình

   mở phiên (ms, không chấm lại BM25)
        │
        ▼
   ┌──────────────────────────────────────────────┐
   │  lưới ảnh ← ffmpeg cắt từ VIDEO GỐC          │
   │      👍 / 👎        ✓ Chốt                    │
   └──────┬───────────────────────┬───────────────┘
          │ click                 │ ghim lên dòng 1
          ▼                       │
   ⑧ ROCCHIO RF                   │
   q′ = norm(α·q₀ + β·c⁺ − γ·c⁻)  │   ~35 ms/vòng
   prototype: cửa sổ ±1,5s        │   drift guard cos(q₀,q′)
          │                       │
          ▼                       │
   emb @ q′  TOÀN 609.476 khung   │   ← không phải rerank danh sách cũ
   + text_rrf có sẵn → ③ lại      │
          │                       │
          └───────► xếp hạng mới ─┘


C ─ XUẤT ─ write_task_csv → pack_submission_zip

   KIS/QA:  chốt + xếp hạng lấp đủ 100 dòng
   TRAKE:   N event × ứng viên → ④ DANTE ghép chuỗi theo thời gian
   Q&A:     ⑥ Qwen2.5-VL đọc khung → đáp án, bọc ngoặc kép
                                │
                                ▼
                    submission/*.csv → submission.zip
```

Toàn bộ đường chạy xếp theo **HẠNG** — không có chuẩn hoá z ở bất kỳ tầng nào.

---

## Công nghệ

| tầng | công nghệ | vai trò |
|---|---|---|
| ⓪ | **jina-clip-v2** (0,9B, 89 ngôn ngữ) | nhúng chung ảnh–chữ; lưu 1024, tra 512. **Ghim** `e10d47f5…` + mã `39e6a55a…` |
| ① | tách mốc `E1:`/`E2:` · `declarativize` | một đề → N probe. Q&A: câu hỏi → câu **mô tả**. TRAKE: mỗi event thành **một câu KIS** |
| ① | mở rộng truy vấn | đọc thư mục `.txt`; **2 bản OCR + 2 bản ASR** mỗi đề, không gọi LLM lúc thi |
| ② | jina-clip · **BM25** ×2 | thị giác **1 run** · OCR **3 run** · ASR **3 run** |
| ③ | **`hierarchical_rrf`** | `beta` gộp expansion TRONG modality, `alpha` gộp modality — cả hai **ĐỀU** |
| ⑤a | **`fused_pool`** | rổ = top-K của điểm ③ |
| ④ | **DANTE** `k_best_alignments` | DP thứ tự thời gian `O(N·T)`; `λ=0`, `min_gap=30` |
| ⑥ | **Qwen2.5-VL** @ Modal | đề Q&A: khung + OCR + lời nói → `answer` |
| ⑦ | `write_task_csv` + `pack_submission_zip` | CSV đúng thể lệ · cổng Tầng 0 (δ) · đọc-ngược-tự-kiểm |
| ⑧ | **Rocchio RF** `src/feedback/` | `q′ = norm(α·q₀ + β·c⁺)`, tìm lại **toàn** chỉ mục |

**Vì sao thị giác giữ đúng MỘT run** dù expansion thị giác đã sinh sẵn: ⑧ Rocchio chỉ
cập nhật run thị giác **gốc** — bản mở rộng là câu khác, không có `q₀` để cập nhật. Cho
thị giác 3 run thì một cú click chỉ động vào 1/3 trọng số modality đó, làm loãng đúng cơ
chế đã chứng minh. Bật lại bằng `--visual-expansion` nếu muốn đo.

BM25 **cố tình không tự bỏ dấu** (`đồng`/`động`). `k1`/`b` riêng từng nguồn:
OCR `0,6 / 0,9` · ASR `1,5 / 0,0`.

---

## Tiến trình thật — đo được, không ước lượng

| bước | thời gian | ghi chú |
|---|---|---|
| ⓪ nạp chỉ mục (`emb.npy` 1,25 GB + BM25) | **71–85 s** | chỉ khi bật server / chạy lô |
| ① mã hoá 132 đoạn trên MPS | **39 s** | tại máy, `$0`, có ghim revision |
| ②③ chấm 24 đề × 7 run trên Modal | **987 s** | 8 CPU · 32 GB |
| A · chuẩn bị lô 24 đề → **33 phiên** | **405 s** | TRAKE tách theo event: 21 + 12 |
| B · **một vòng click** | **7–68 ms** | hợp đồng đòi < 500 ms ⟹ dư 7 lần |
| ⑤c rerank API 1.440 khung | 1.698 s | API chấm được 1.210/1.440 = 84% |
| ⑥ Qwen sinh 3 đáp án Q&A | ~120 s | ảnh cắt từ video gốc |

Đĩa: `data/rf_sessions/` **154 MB cho 33 phiên** (`text_rrf` giữ nguyên 609.476 chiều —
cắt xuống 100 dòng thì Rocchio tụt thành rerank danh sách cũ).

**Vì sao tách A và B.** Vòng 0 tốn ~29 s/đề vì BM25 quét 609.476 khung. Bắt operator
chờ từng câu là đốt đúng thứ khan hiếm nhất của 2h30. Chạy lô trước thì lúc mở UI mọi
đề đã có top, và mỗi cú click chỉ tốn mili-giây.

**Vì sao ⑧ chạy tại máy, không Modal.** Mỗi lời gọi Modal trả phí cố định ~200 s nạp
chỉ mục; hợp đồng đòi vòng feedback < 500 ms. Chỉ mục nằm sẵn trong RAM tại máy thì
vòng click là một phép nhân ma trận cục bộ.

Hạ tầng **Modal** cho ②③ theo lô và ⑥: app `aic-query`, volume `aic-query-index`
1,4 GB (gọn từ 4,3 GB — OCR 2,9 GB → 128 MB vì BM25 chỉ cần một trường chữ).

---

## Trọng số

```
③ alpha (modality)   visual 1/3 · ocr 1/3 · asr 1/3      ĐỀU
③ beta  (expansion)  đều trong từng modality, 3 run ⟹ 1/3
⑤ giai đoạn          fused4 1,0 · crop 0,0 · vlm 0,25
```

`RRF_K = 60` · `POOL_CAP = 300` · `VLM_TOP_K = 160` · `TRAKE_K_PER_VIDEO = 1`

Chưa quét lại sau khi bỏ z-norm: `vlm = 0,25` (đo dưới z-norm, khác đơn vị), `RRF_K`,
`POOL_CAP` (nay là phép cắt chủ động, không còn là chốt an toàn).

---

## Định dạng bài nộp

Theo "Hướng dẫn nộp bài sơ tuyển". Thể lệ cho **3 lượt mỗi gói**, và *nộp sai định dạng
vẫn tính một lượt* — mọi lỗi ở tầng này đều **câm**.

| luật | cài ở đâu |
|---|---|
| một `.csv` mỗi đề, tên khớp tên đề | `write_task_csv` |
| **hậu tố** `kis`/`qa`/`trake` quyết định loại | `task_type_from_filename` |
| ≤100 dòng · không header · UTF-8 · dấu phẩy | `csv.writer(lineterminator="\n")` |
| ngoặc kép khi đáp án có `,` `"` xuống dòng | `QUOTE_MINIMAL` |
| đáp án ≤100 ký tự, bộ chấm **không tự trim** | `clean_answer` — cắt kèm ghi chú |
| tên video không `.mp4` · `frame_id` số nguyên | `format_task_csv` |
| zip phải chứa thư mục `submission/` | `pack_submission_zip` |

`write_task_csv` **đọc ngược file bằng `csv.reader`** sau khi ghi — phép kiểm duy nhất
chạy trên chuỗi byte sẽ nộp đi. `pack_submission_zip` chỉ nhận `.csv`, nên `_report.json`
và `_rerank_scores.npz` không lọt vào bài nộp.

---

## Kết quả

### ⑧ Rocchio RF — **có tác dụng, và cần một cú click thật**

249 câu có đáp án (`gen299`), bootstrap **bắt cặp** 4000 lần. Chỉ tính trên **64 câu
operator thật sự click được** — tức khung đúng nằm trong top-100, điều kiện §11.1 của
tài liệu:

| β | ΔMRR | KTC95 | thắng/thua |
|---|---|---|---|
| 0,4 | +0,1770 | `[+0,1035, +0,2575]` | 34/6 |
| 0,65 | +0,2455 | `[+0,1601, +0,3313]` | 38/2 |
| **0,8** ★ | **+0,2557** | `[+0,1680, +0,3446]` | 38/3 |
| 1,0 | +0,2643 | `[+0,1739, +0,3552]` | 39/2 |

`MRR 0,3967 → 0,6524`. Mọi KTC nằm **trọn trên 0**.

Chọn `β = 0,8` chứ không `1,0`: chênh `+0,0086` nằm sâu trong nhiễu, còn `β=1,0` nghĩa là
prototype nặng **ngang** câu hỏi gốc — mất neo khi click nhầm.

🔴 **Bản tự động (không có người click) thì vô dụng.** Pseudo-relevance feedback lấy
top-`m` làm positive: mọi cấu hình có KTC **chứa 0**, tốt nhất `+0,0017`. Giá trị nằm ở
cú click của người, không ở thuật toán chạy tự động.

### Nút thắt thật — **truy hồi, không phải xếp hạng**

Hạng của khung đúng trên 249 câu:

| nguồn | R@10 | R@100 | R@1000 | trung vị hạng |
|---|---|---|---|---|
| thị giác | 9,6% | 25,3% | 47,8% | 1.264 |
| OCR | 10,4% | 16,5% | 22,9% | 113.584 |
| ASR | 6,4% | 19,3% | 32,5% | 13.584 |
| **hợp ③** | **19,7%** | **41,0%** | 57,4% | **381** |

Dung hợp tốt hơn **mọi** nguồn đơn — nó không chôn kết quả tốt. Nhưng **147 câu** vẫn
rớt khỏi top-100, và trong đó:

```
14 câu (9,5%)  có ít nhất một nguồn đưa được vào top-100  ⟹ chữa được bằng dung hợp
133 câu (90,5%) mọi nguồn đều mù                          ⟹ lỗi TRUY HỒI
```

**Trần của việc tune trọng số là 5,6% số câu.** Và chỉ **1/249** câu có khung đúng vắng
mặt khỏi chỉ mục — nên không phải lỗi cắt khung.

### TRAKE — chưa cấu hình nào chứng minh được

50 câu, chấm nguyên văn công thức thể lệ. Nền `gap=1 · k=1` cho `Final 0,0114`; cấu hình
tốt nhất `gap=30 · k=10` cho `0,0186`, nhưng **mọi KTC đều chứa 0** — 46/50 câu ăn 0 điểm
ở mọi cấu hình nên không đủ lực thống kê. Trần do mật độ khung là **0,4273**.

**λ = 0 thì được xác nhận**: mọi `λ > 0` bằng hoặc tệ hơn ở **cả 45 cấu hình**.

---

## Chạy

**Ngày thi — hai bước, bước A chạy TRƯỚC khi nhận đề:**

```bash
# A · chuẩn bị lô: vòng 0 cho cả gói, ghi phiên xuống đĩa
python scripts/rf_prepare.py --dir <thư-mục-đề>

# B · công cụ click (nạp ~75 s — bật SỚM, đừng bật lúc nhận đề)
python scripts/rf_server.py --port 8000     # → http://127.0.0.1:8000
```

Trong UI: chọn phiên → **Mở phiên** → 👍 khung gần đúng → **✓ Chốt** khung tốt nhất →
**Xuất CSV** → **Đóng gói .zip**. `Reset`/`Undo` có sẵn; mọi thao tác ghi nhật ký nên
server chết cũng không mất việc.

**Đường chạy tự động (không có người click):**

```bash
python scripts/run_thunghiem_modal.py --out submission_p1      # ①②③⑤a Modal, ⑤c④⑥⑦ máy
python scripts/finalize_submission.py --sub <thư-mục>/submission  # chỉ sinh đáp án + ghi lại
```

**Nghiên cứu:**

```bash
python scripts/research/eval_query_sets.py            # Recall/MRR trên 299 đề
python scripts/research/tune_rocchio.py               # quét lưới Rocchio §11.3
```

Đầu ra `submission/*.csv` + `submission.zip`. Cột thứ hai trở đi là **`frame_idx`** —
số khung THẬT trong `.mp4` gốc. Đã kiểm ba lớp: `δ = frame_idx − round(pts_time × fps)`
bằng 0 trên **609.476/609.476** dòng; `ffprobe` trên video gốc cho khung đầu
`pts_time=0.000000` ⟹ 0-based cùng quy ước; `max(frame_idx) < nb_frames` trên 40/40 mẫu.

---

## Clone về rồi chạy

Kho mang **mã và bằng chứng**, không mang dữ liệu nặng. Sau khi clone cần ba thứ:

| cần | ở đâu | dùng cho |
|---|---|---|
| `data/embed/` — chỉ mục 609.476 × 1024 | dựng bằng `scripts/index/` | mọi thứ |
| `data/OCR/ocr.jsonl` · `data/ASR/` | tiền xử lý của đội | ② BM25 |
| `data/keyframes/*.zip` — 3 kho ảnh, 29 GB | đội tự cắt | hiển thị trong UI |

Không có kho ảnh thì UI tự lùi về `ffmpeg` cắt từ `data/video/` — chậm hơn ~300 lần
nhưng vẫn chạy.

Bản mở rộng cho bộ 299 là file **dẫn xuất**; nguồn JSON đã có trong kho:

```bash
python - <<'EOF'
import json; from pathlib import Path
out = Path("expansions_gen299"); out.mkdir(exist_ok=True)
for src, tag in (("qo","ocr1"), ("qa","asr1"), ("qv1","vis1"), ("qv2","vis2")):
    for r in json.load(open(f"data/research/expansions/gen299_{src}.json",
                            encoding="utf-8"))["queries"]:
        if r["query"].strip():
            (out / f"{r['id']}.{tag}.txt").write_text(r["query"].strip(), encoding="utf-8")
EOF
```

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 877 test — phải xanh trước khi tin bất cứ số nào
```

---

## Bố cục

```
scripts/
  rf_prepare.py              A · vòng 0 theo lô → data/rf_sessions/
  rf_server.py               B · API §6 + UI §7, vòng click ~35 ms
  run_thunghiem_modal.py     đường chạy TỰ ĐỘNG: ①②③⑤a Modal · ⑤c④⑥⑦ máy
  modal_query.py             app Modal: encode_text · score · qa_answer · rocchio_sweep
  finalize_submission.py     sinh đáp án Q&A + ghi lại CSV, KHÔNG chạy lại truy xuất
  research/                  công cụ THÍ NGHIỆM, ngoài đường chạy thi
src/
  feedback/rocchio.py        ⑧ q′ = norm(α·q₀ + β·c⁺ − γ·c⁻) — 11 test tính chất
  feedback/session.py        trạng thái phiên §4.1 · state RIÊNG từng event §8
  feedback/store.py          nhật ký nối thêm — chết rồi khôi phục đúng trạng thái
  retrieval/score_matrix.py  ③⑤ hierarchical_rrf — RRF hai tầng
  retrieval/pool.py          ⑤a fused_pool — top-K của điểm đã hợp
  retrieval/sources.py       ② ba nguồn + `covered`
  submission/kbest.py        ④ k-best của cùng DP — KIS là N=1
  submission/writer.py       ⑦ CSV theo thể lệ + cổng Tầng 0 + đóng gói .zip
  scoring/rscore.py          R-Score và Final, nguyên văn thể lệ
data/
  embed/                     chỉ mục phẳng 609.476 × 1024 float16
  rf_sessions/<id>/          meta.json · vectors.npz · events.jsonl
  video/                     873/873 .mp4 gốc — nguồn ảnh DUY NHẤT đúng
```
