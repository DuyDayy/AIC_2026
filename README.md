# AIC 2026 Multimodal Video Retrieval

Hệ truy xuất video cho Textual KIS, Q&A và TRAKE. Tầng truy vấn kết hợp
Jina-CLIP-v2, OCR, object, ASR, Qwen2.5-VL và dynamic programming để sinh CSV chứa
`frame_idx` của video gốc.

> Repository chỉ chứa mã nguồn và một số manifest/report nhỏ. Keyframe, embedding,
> OCR, ASR, object data và các submission chạy thật không nằm trong Git.

## Kiến trúc đang chạy

```text
query text
  → probe / tách event
  → visual cosine + BM25(OCR, object, ASR)
  → z-score fusion
  → union top-40 riêng từng nguồn
  → fused score + Qwen VLM rerank
  → DANTE/k-best
  → temporal spread
  → CSV
```

| Thành phần | Cấu hình hiện tại |
|---|---|
| Visual | Jina-CLIP-v2; index lưu 1024 chiều, truy xuất ở 512 chiều |
| Text sources | BM25 trên OCR, object và ASR |
| Fusion | visual `1.0`, object `0.0891`, OCR `0.1322`, ASR `0.2128` |
| Candidate pool | top-40 mỗi nguồn; tối đa 10 frame/video/nguồn; cap cấu hình 200 |
| Rerank | `1.0 × fused4 + 0.25 × VLM`; crop reranker đang tắt |
| VLM | Qwen2.5-VL-7B, chấm top-30 candidate của KIS/QA |
| Temporal ranking | DANTE/k-best, `lambda = 0` |
| Output budget | tối đa 100 dòng/query |
| Baseline spread | 7; mặc định CLI hiện là 1 |

TopK riêng chỉ quyết định candidate nào được vào pool. Thứ hạng cuối vẫn dựa trên fused
score và VLM. Cấu hình 512 chiều được giữ vì A/B development không phân biệt được với
1024 chiều trong khi ma trận làm việc nhỏ bằng một nửa.

## Yêu cầu

- Python 3.11.
- Tài khoản Modal có quyền dùng GPU.
- Dữ liệu runtime đúng layout bên dưới.
- RAM đủ để nạp embedding và dựng ba BM25 index.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
modal setup
```

## Dữ liệu runtime

```text
data/
├── embed/
│   ├── emb.npy
│   ├── ids.npy
│   ├── frame_idx.npy
│   └── ranges.json
├── OCR/ocr.jsonl
├── ASR/*/results/*.json
├── objects-full/<video_id>/<n>.json
└── Framme/*/metadata/*.csv
```

Ảnh keyframe phải nằm trong một trong các `KEYFRAME_ROOTS` khai báo tại
`scripts/run.py`. Các nguồn được join bằng `(video_id, n)`; đầu ra luôn dùng
`frame_idx`, không dùng số thứ tự keyframe `n`.

## Chuẩn bị truy vấn

Mỗi truy vấn là một file `.txt` trong `queries/`; tên file trở thành `query_id`.

Loại truy vấn được nhận diện theo thứ tự:

1. Tên file chứa `trake`, `qa` hoặc `kis`.
2. Nội dung có các marker `E1:`, `E2:`, ... được xem là TRAKE.
3. Trường hợp còn lại là KIS.

Ví dụ TRAKE:

```text
Một đoạn dạy nấu ăn. E1: Dầu sôi trong chảo. E2: Vợt lưới vớt thức ăn ra.
```

## Chạy

Chạy cấu hình tương ứng với `submission_v2`:

```bash
modal run scripts/run.py --dir queries --out submission --spread 7
```

Các chế độ giảm chi phí:

```bash
# Bỏ Qwen VLM, giữ retrieval đa nguồn
modal run scripts/run.py --dir queries --out submission --spread 7 --vlm-top-k 0

# Bỏ rerank
modal run scripts/run.py --dir queries --out submission --spread 7 --no-rerank

# Bỏ OCR, object và ASR; VLM vẫn chạy nếu không thêm --no-rerank
modal run scripts/run.py --dir queries --out submission --spread 7 --light
```

Nên truyền `--spread` tường minh: code mặc định là `1`, còn baseline được đánh giá dùng
`7`. Vector query được cache tại `data/embed/query_cache.npz`.

## Đầu ra

Mỗi query sinh một file CSV không có header:

```text
KIS:    video_id,frame_idx
QA:     video_id,frame_idx,answer
TRAKE:  video_id,frame_idx_1,frame_idx_2,...
```

Các frame TRAKE thuộc cùng video và tăng theo thời gian. Một run còn sinh:

- `_report.json`: cấu hình và top-1 của từng query.
- `_rerank_scores.npz`: điểm trung gian trong pool khi bật rerank.

## Kết quả development hiện tại

`submission_v2` đã được chấm lại trực tiếp trên 100 KIS development query, với cửa sổ
proxy rộng 9 frame đặt quanh keyframe được gán nhãn:

| R@1 | R@5 | R@20 | R@50 | R@100 | Final |
|---:|---:|---:|---:|---:|---:|
| 0.44 | 0.44 | 0.66 | 0.78 | 0.80 | 0.624 |

Bộ query này đã được dùng để chọn cấu hình. Đây là development proxy, không phải
held-out score hoặc điểm official. QA chưa có benchmark đủ lớn; TRAKE hiện chỉ có sáu
query thử nghiệm nên chưa có kết luận định lượng đáng tin cậy cho hai task này.

## Dựng lại embedding index

```bash
# Đo throughput và chi phí trước
modal run scripts/index/2_encode_frames.py --benchmark 400

# Mã hóa toàn bộ keyframe
modal run scripts/index/2_encode_frames.py

# Tải shard về và dựng flat index tại data/embed
modal volume get aic-data-vol embed-jina-v2 /tmp/emb/
python scripts/index/3_build_index.py --src /tmp/emb/embed-jina-v2

# Ghi provenance của model và artifact
modal run scripts/index/4_write_manifest.py
```

Manifest hiện nằm tại `artifacts/embed/embed/manifest.json`.

## Mã nguồn chính

```text
scripts/run.py                    pipeline truy vấn và sinh CSV
scripts/index/                    mã hóa, dựng index và manifest
src/ingestion/vector_index.py     flat exact vector index
src/retrieval/sources.py          visual, OCR, object và ASR
src/retrieval/score_matrix.py     z-normalization và fusion
src/retrieval/pool.py             candidate union
src/retrieval/rerank.py           crop/VLM scores
src/retrieval/dante.py            temporal dynamic programming
src/submission/kbest.py           KIS/TRAKE k-best
src/submission/coverage.py        temporal spread
```

## Giới hạn hiện tại

- Clone sạch chưa chạy được nếu không có data runtime bên ngoài Git.
- Development score chưa phải held-out hoặc official evaluation.
- QA chỉ đọc top-1 anchor rồi dùng cùng answer cho toàn bộ candidate output.
- TRAKE chưa chạy VLM reranker.
- Spread có thể phát raw frame ID chưa được decode và chấm trực tiếp.
- Model revision và query-cache provenance chưa được khóa hoàn toàn ở runtime.
- Production hiện ghi CSV trực tiếp, chưa đi qua submission validator.

Kế hoạch tối ưu, protocol đánh giá và tiêu chí promote nằm tại
[OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md).
