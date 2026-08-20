# Tầng truy vấn — AIC 2026

Hệ truy xuất video đa phương thức, vòng sơ tuyển. Tiền xử lý (khung hình · ASR · OCR) đã
xong; kho này là **tầng truy vấn**: từ đề bài ra bài nộp.

Vận hành ngày thi: [`RUNBOOK.md`](RUNBOOK.md) · Tối ưu: [`OPTIMIZATION_PLAN.md`](OPTIMIZATION_PLAN.md)

```
⓪ mã hoá offline   173.426 khung → lưu 1024 chiều, tra ở 512
       │
đề → ① PROBE + mở rộng → ② 7 run chấm → ③ DUNG HỢP  RRF hai tầng
     tách mốc E1/E2       1 thị giác      beta  expansion trong modality
     đọc expansions/      3 OCR           alpha modality với nhau
                          3 ASR                     │
                                                    ▼
             ⑦ nộp ←── ④ DANTE ←── ⑤c RERANK ←── ⑤a RỔ
             CSV+zip     k-best      VLM          top-300 của điểm ③
```

Toàn bộ đường chạy xếp theo **HẠNG** — không có chuẩn hoá z ở bất kỳ tầng nào.

---

## Công nghệ

| tầng | công nghệ | vai trò |
|---|---|---|
| ⓪ | **jina-clip-v2** (0,9B, 89 ngôn ngữ) | nhúng chung ảnh–chữ; Matryoshka lưu 1024, tra 512 |
| ① | tách mốc `E1:`/`E2:` · `declarativize` | một đề → N probe. Q&A: câu hỏi → câu **mô tả** |
| ① | mở rộng truy vấn | đọc `expansions/` — 2 bản OCR + 2 bản ASR mỗi đề, **không gọi LLM lúc thi** |
| ② | jina-clip · **BM25** ×2 | thị giác · OCR · lời nói. Mỗi modality văn bản chấm 3 lần |
| ③ | **`hierarchical_rrf`** | `beta` gộp expansion TRONG modality, `alpha` gộp modality |
| ⑤a | **`fused_pool`** | rổ = **top-300 của điểm ③**, bộ lọc cứng |
| ⑤c | **Qwen2.5-VL-7B** | `P(khung khớp)` = softmax hai token `1`/`0`; hợp bằng RRF tầng ba |
| ④ | **DANTE** k-best | DP thứ tự thời gian `O(N·T)`; **KIS = TRAKE với N=1** |
| ⑥ | Qwen2.5-VL | chỉ đề Q&A: khung + OCR + lời nói → `answer` |
| ⑦ | CSV theo thể lệ + đóng gói `.zip` | một khung mỗi mốc, 100 mốc |

BM25 **cố tình không tự bỏ dấu** (`đồng`/`động`, `cán`/`căn`/`cân`). `k1`/`b` riêng từng
nguồn: OCR `0,6 / 0,9` · ASR `1,5 / 0,0`.

Hạ tầng **Modal**, GPU A10G. `.map()` nhanh **31×** so với `.remote()` tuần tự — nghẽn là
đường truyền, không phải GPU.

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

Chấm trên `data/eval/bench_kis_gt.json` (100 câu). **Không so ngang** với bảng `Final`
chấm theo luật BTC — script ở `scripts/research/`, số ở `data/research/`.

### E1 — nguồn đơn và dung hợp, `R@100`

`load_frame_ms()` từng trả `{}` vì thư mục metadata bị xoá, nên **ASR phủ 0/173.426 khung
và ghi 0 điểm cho mọi truy vấn** — không nổ, không cảnh báo. Sau khi vá (lùi về
`data/OCR/ocr.jsonl`, phủ 173.426/173.426 khoá) ASR phủ **96,9%**.

| | R@100 trước vá | R@100 sau vá |
|---|---|---|
| asr only | 0,00 | **0,58** |
| asr + `qa` | 0,00 | **0,62** |
| visual + asr | 0,65 | **0,85** |
| dung hợp cả ba | 0,73 | **0,91** |

ASR là nguồn đơn mạnh **thứ hai** (0,58 so với OCR 0,46).

### E3 — ma trận cấu trúc phân cấp, `R@100`

| cấu hình | R@100 |
|---|---|
| V + O gốc + A gốc *(không QE)* | 0,91 |
| V + O `qo` + A `qa` *(bỏ bản gốc)* | 0,93 |
| V + (O gốc+`qo`) + A gốc | 0,92 |
| V + O gốc + (A gốc+`qa`) | 0,92 |
| **V + (O gốc+`qo`) + (A gốc+`qa`)** ★ | **0,93** |
| 5 run PHẲNG *(không phân cấp)* | 0,89 |

Phân cấp hơn phẳng; giữ bản gốc bên cạnh expansion hơn bỏ nó; QE cho **cả hai** nhánh văn
bản mới đạt đỉnh. **E4** (tune trọng số toàn cục) là **NO-GO**: ΔMRR `−0,0398`, KTC95
`[−0,0783, −0,0056]` — trọn dưới 0.

---

## Chạy

Mỗi truy vấn là **một file `.txt`**, tên file thành `query_id` và quyết định loại đề.

```bash
modal run scripts/run.py --dir queries --out submission
```

```bash
modal run scripts/run.py --vlm-top-k 0    # bỏ ⑤c, tiết kiệm ~4 phút
modal run scripts/run.py --light          # chỉ thị giác, máy thiếu RAM
python scripts/eval/score_submission.py --sub submission --gt <gt.json>
```

Đầu ra `submission/{id}.csv` + `submission.zip`. Cột thứ hai trở đi là **`frame_idx`** —
số khung THẬT, không phải `n`; đo được 0/173.426 khung có hai giá trị bằng nhau.

Ngân sách **~3,5 phút / 35 câu** không bật ⑤c, **~7 phút** có bật (~5% của 2h30). Phí cố
định 80 giây nạp chỉ mục trả **mỗi lần gọi** — gom lô.

---

## Bố cục

```
scripts/run.py                 đường chạy tổng — ①②③④⑤⑥⑦
scripts/index/                 dựng lại chỉ mục, chạy theo số thứ tự
scripts/eval/
  score_submission.py          chấm ĐÚNG LUẬT BTC, quét L, bảng R@k
  compare_arch.py              so kiến trúc từ bản lưu ⑤
  make_queries.py              sinh bộ eval giữ kín
src/
  ingestion/jina_encoder.py    Matryoshka: CẮT rồi mới chuẩn hoá
  ingestion/vector_index.py    chỉ mục phẳng, mang cả `n` lẫn `frame_idx`
  retrieval/probe.py           ①  tách mốc; Q&A → câu mô tả
  retrieval/sources.py         ②  ba nguồn + `covered`
  retrieval/bm25.py                BM25, cố tình KHÔNG tự bỏ dấu
  retrieval/score_matrix.py    ③⑤ hierarchical_rrf — RRF hai tầng
  retrieval/pool.py            ⑤a fused_pool — top-K của điểm đã hợp
  retrieval/rerank.py          ⑤bc mảnh cắt + VLM
  submission/kbest.py          ④⑦ k-best của cùng DP — KIS là N=1
  submission/writer.py         ⑦  CSV theo thể lệ + đóng gói .zip
  scoring/rscore.py                R-Score và Final, nguyên văn thể lệ
scripts/research/              QAFuse E0–E4 — công cụ THÍ NGHIỆM, ngoài đường chạy thi
data/research/                 kết quả E0–E4
```
