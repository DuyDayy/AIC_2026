# Kế hoạch tối ưu toàn bộ hệ thống AIC 2026

> Trạng thái: kế hoạch triển khai
>
> Baseline tham chiếu: `submission_v2`
>
> Nguyên tắc: sửa correctness và khả năng tái lập trước; tối ưu candidate recall trước
> reranker; chỉ thay baseline khi thắng trên tập blind với cùng ngân sách.

## 1. Kết luận kiến trúc cần chốt

Không chọn tuyệt đối giữa **fuse score** và **hợp topK riêng từng nguồn**. Hai kỹ thuật
giải hai bài toán khác nhau:

- Fuse score giữ precision và tạo thứ hạng nền đáng tin cậy.
- TopK riêng từng nguồn bảo vệ candidate chuyên biệt OCR, ASR hoặc object khỏi bị trọng
  số fusion nhỏ làm biến mất.
- VLM chỉ có thể rerank candidate mà nó được nhìn thấy. Vì vậy shortlist đưa vào VLM
  cũng phải có phần `fused core` và phần `source rescue`.

Kiến trúc mục tiêu:

```text
visual / OCR / ASR / object scores
                 │
                 ├── weighted fusion ──────── fused core ──────┐
                 │                                              │
                 └── top riêng từng nguồn ── rescue candidates ├── pool cố định
                                                                │
                                        fused + rescue selector ├── VLM shortlist
                                                                │
                                         fused score + VLM score└── final ranking
                                                                         │
                                                        temporal refine / output
```

Hướng mặc định để thử đầu tiên là **Hybrid admission + Hybrid VLM shortlist +
Fused/VLM final ranking**.

## 2. Baseline phải khóa bất biến

Baseline lấy từ `submission_v2/_report.json`:

| Thành phần | Giá trị |
|---|---:|
| Embedding dimension | 512 |
| Visual weight | 1.0000 |
| Object weight | 0.0891 |
| OCR weight | 0.1322 |
| ASR weight | 0.2128 |
| Candidate admission | union top-40 mỗi nguồn |
| Per-source/per-video cap | 10 |
| Pool cap cấu hình | 200 |
| Crop reranker | 0 |
| VLM weight | 0.25 |
| VLM shortlist | top-30 theo fused score trong pool |
| Spread | 7 |
| Output budget | 100 dòng/query |
| Development proxy Final, fixed L=9 | 0.624 |

`0.624` chỉ là development proxy trên bộ 100 query đã được dùng để tuning; không được
dùng làm bằng chứng cuối cùng để promote kiến trúc mới.

Mỗi lần chạy baseline phải lưu nguyên vẹn dưới run ID riêng. Không ghi đè
`submission_v2`, query, GT hoặc artifact cũ.

## 3. Chẩn đoán bước topK hiện tại

Đường chạy hiện tại tại `scripts/run.py` là:

1. Mỗi probe chấm visual, OCR, object và ASR.
2. Tạo fused score trên toàn index.
3. Mỗi nguồn đề cử top-40 riêng, rồi union thành pool khoảng 156 frame.
4. Trong pool, fused score vẫn quyết định thứ hạng nền.
5. Chỉ top-30 theo fused score được VLM nhìn.
6. Final score là `fused4 + 0.25 * VLM` sau chuẩn hóa.

Các lỗ hổng cần xử lý trong experiment:

### 3.1 Candidate specialist được cứu vào pool nhưng không được rerank

Một candidate do OCR/ASR cứu vào pool có thể có fused score thấp. Nó sẽ không lọt top-30
VLM, nên admission tốt hơn không chuyển thành Final tốt hơn.

### 3.2 Candidate joint-evidence có thể bị loại

Một frame không top-40 ở nguồn nào nhưng cùng lúc khá tốt ở nhiều nguồn có thể có fused
score cao. Union thuần sẽ bỏ frame này dù fuse-first giữ được.

### 3.3 Zero-score lexical candidate đang có thể chiếm quota

`SourceScores.covered=True` chỉ nói frame có dữ liệu OCR/ASR/object, không nói query khớp.
Với nguồn lexical, candidate chỉ đủ điều kiện khi:

```text
eligible = covered AND score > 0
```

Nếu không, topK có thể chứa các frame BM25 bằng 0 và tie-break theo vị trí index.

### 3.4 Pool cap chưa phải global budget

Cap hiện áp trong từng probe. Query nhiều probe có thể hợp lại thành pool lớn hơn cap.
Cap phải được áp sau khi hợp mọi probe, khử trùng và refill.

### 3.5 Không được interleave nguồn để làm final ranking

Round-robin topK giữa bốn nguồn đã từng làm Final giảm khoảng `0.143`. TopK riêng chỉ
quyết định ai được vào vòng rerank; final ranking vẫn phải dùng fused score và VLM.

## 4. Ba ngân sách phải tách riêng

Không dùng một chữ `K` cho ba đại lượng khác nhau:

| Ký hiệu | Ý nghĩa | Baseline gần đúng |
|---|---|---:|
| `B_pool` | Số candidate keyframe duy nhất trước rerank | 156 |
| `K_vlm` | Số query-frame thật sự trả phí VLM | 30 |
| `M_emit` | Số semantic anchor được phát trước spread | khoảng 14 với spread 7 |

Mọi A/B phải giữ cố định cả ba đại lượng và output budget 100. Nếu một nhánh dùng nhiều
VLM call hoặc phát nhiều anchor hơn, chênh lệch không còn đến riêng từ admission policy.

## 5. Candidate policy cần triển khai

Nên tạo interface dùng chung, ví dụ:

```python
class CandidatePolicy(Protocol):
    def select(
        self,
        source_scores,
        fused_score,
        budget,
        metadata,
    ) -> CandidatePool: ...
```

`CandidatePool` phải lưu:

- danh sách row đã sắp tất định;
- raw score và rank của từng nguồn;
- fused score;
- provenance nguồn;
- video, shot, keyframe và raw `frame_idx`;
- lý do admission: `fused_core`, `visual_rescue`, `ocr_rescue`, `asr_rescue` hoặc
  `object_rescue`.

### 5.1 Policy F — Fuse-first

```text
FUSE(B):
    rank tất cả frame bằng fused_score
    áp cùng diversity/cap rule
    lấy đúng B candidate duy nhất
```

Ưu điểm: precision cao, đơn giản và nhanh.

Rủi ro: bỏ candidate chỉ có bằng chứng mạnh ở một nguồn có trọng số nhỏ.

### 5.2 Policy U — Union-first

```text
UNION(B, quota):
    tạo ranked list riêng cho từng nguồn
    loại zero-score khỏi nguồn lexical
    lấy theo quota và hợp candidate
    sau dedup, tiếp tục duyệt các list cho tới đúng B
    nếu nguồn hết positive hit, backfill bằng fused list
```

Không được dừng ở `sum(quota)` trước dedup, vì policy có overlap lớn sẽ vô tình dùng ít
ngân sách hơn policy khác.

### 5.3 Policy H — Hybrid

```text
HYBRID(B, K_fused, quota):
    pool = top K_fused của fused list
    thêm candidate rescue của từng nguồn nếu chưa có
    bảo toàn quota tối thiểu của nguồn còn positive hit
    backfill bằng fused list cho tới đúng B
    áp global diversity/cap và refill lần cuối
```

Grid đầu tiên với `B_pool=160`:

| Cấu hình | Fused core | Rescue visual | Rescue OCR | Rescue ASR | Rescue object |
|---|---:|---:|---:|---:|---:|
| H40 | 120 | 10 | 10 | 10 | 10 |
| H80 | 80 | 20 | 20 | 20 | 20 |
| H120 | 40 | 30 | 30 | 30 | 30 |

Đây chỉ là ba điểm kiến trúc thô để tìm vùng tốt; không tinh chỉnh từng slot trước khi có
held-out validation.

### 5.4 Multi-probe

Với query nhiều probe, không cấp lại toàn bộ `B_pool` cho mỗi probe. Trước hết gộp ở cấp
query:

```text
source_score_s(frame) = max score_s(frame) qua các probe
fused_score(frame)    = max fused_score(frame) qua các probe
```

Sau đó candidate policy chỉ được dùng đúng một global budget. Với TRAKE, cần lưu thêm
per-event admission để đo event coverage; không làm mất ma trận score của từng probe.

### 5.5 Diversity và per-video cap

Để so admission policy công bằng:

- Giữ `U-v2-exact` làm reference tái tạo hành vi hiện tại.
- Trong A/B chính, áp cùng một global per-video rule sau khi dựng pool.
- Sau khi cap loại candidate, refill cho tới đủ `B_pool`.
- Chỉ quét `none/20/10/5` sau khi đã chọn được policy admission tốt.
- Không thay admission policy và cap trong cùng một phép kết luận.

## 6. VLM shortlist policy

Đây là experiment độc lập với candidate admission.

### 6.1 VLM-F

```text
top K_vlm theo fused score trong pool
```

Đây là baseline hiện tại.

### 6.2 VLM-U

```text
quota riêng từ các ranked list nguồn trong pool
dedup + refill cho đủ K_vlm
```

Policy này dùng để đo cực union; không dùng rank nguồn làm final ranking.

### 6.3 VLM-H

Seed config cho `K_vlm=30`:

| Thành phần | Số frame |
|---|---:|
| Fused core | 18 |
| Visual rescue | 3 |
| OCR rescue | 3 |
| ASR rescue | 3 |
| Object rescue | 3 |

Sau dedup hoặc khi một nguồn hết positive hit, backfill bằng fused list cho đủ 30.

Không chạy GPU lại cho từng cấu hình. Lấy hợp tất cả query-frame cần bởi các policy còn
trên Pareto frontier, chấm mỗi cặp đúng một lần rồi replay offline.

Cache key bắt buộc:

```text
(model_sha, prompt_sha, query_hash, video_id, frame_idx)
```

## 7. Ma trận thí nghiệm và successive halving

Không chạy tích Descartes đầy đủ ngay từ đầu.

### Vòng 1 — Candidate oracle, chi phí GPU bằng 0

```text
B_pool ∈ {40, 80, 120, 160, 200}
policy ∈ {F, U, H40, H80, H120}
K_vlm = 0
spread cố định
```

Giữ tối đa 3–4 cấu hình nằm trên Pareto frontier của recall, oracle score, latency và
redundancy.

### Vòng 2 — VLM selector

```text
K_vlm ∈ {10, 20, 30, 50}
selector ∈ {VLM-F, VLM-U, VLM-H}
candidate policy ∈ Pareto candidates từ vòng 1
```

Chấm VLM trên superset một lần rồi replay tất cả cấu hình.

### Vòng 3 — Final ranking

- Khóa candidate policy và VLM selector.
- Giữ `fused4=1`, `crop=0`, `VLM=0.25` làm baseline.
- Chỉ quét một grid nhỏ cho VLM weight.
- Dùng nested/grouped CV trên validation, không lấy đỉnh trực tiếp toàn tập.
- Chỉ tối ưu spread sau khi admission và reranking đã khóa.

### Vòng 4 — Locked test

- Freeze đúng một challenger và baseline.
- Ghi hash config/code/model/index trước khi mở test.
- Chạy test đúng một lần.
- Không sửa config theo failure của locked test.

## 8. Candidate funnel và metric bắt buộc

### 8.1 Trước rerank

- Correct-video recall@`B_pool`.
- True-interval hit@`B_pool`.
- Oracle Final nếu reranker hoàn hảo trên chính pool.
- Best rank của target candidate theo fused score.
- Số candidate đúng trong pool.
- Số video và shot duy nhất.
- Redundancy theo video/shot/frame.
- Overlap/Jaccard giữa các nguồn.
- Unique contribution của nguồn `s`:

```text
Hit(all sources) - Hit(all sources except s)
```

- TRAKE event coverage và oracle path score.

Không dùng spread khi đo candidate recall. Spread hình học có thể che lỗi admission.

### 8.2 Qua VLM

Với mỗi query phải ghi funnel:

```text
target video có trong pool?
    └── true interval có candidate?
          └── candidate đúng có được VLM nhìn?
                └── vào top-1/5/20/50/100 cuối?
```

Metric:

- VLM capture rate.
- Rank promotion/demotion.
- `OracleFinal(pool) - Final thực`.
- `P(hit top-k | pool đã có candidate đúng)`.
- Số VLM call và tỷ lệ cache hit.

### 8.3 End-to-end

- Official `R@1`, `R@5`, `R@20`, `R@50`, `R@100`.
- Final Score.
- Per-task và per-modality score.
- Paired win/tie/loss.
- P50/P95 latency.
- GPU-seconds và chi phí/query.
- Số output thiếu, duplicate hoặc invalid.

## 9. Bộ eval chống leakage

### 9.1 Vai trò bộ 100 hiện tại

Chỉ dùng để:

- smoke test;
- regression test;
- phát hiện bug;
- so hướng kiến trúc sơ bộ.

Không dùng nó để tuyên bố cấu hình thắng vì query và GT đã tham gia nhiều vòng tuning.

### 9.2 Dataset mục tiêu

| Split | KIS | QA | TRAKE | Vai trò |
|---|---:|---:|---:|---|
| `blind-validation-180` | 120 | 30 | 30 | Chọn cấu hình |
| `locked-test-180` | 120 | 30 | 30 | Mở đúng một lần |

Yêu cầu:

- Video và query family không giao nhau giữa hai split.
- Phân tầng theo L21–L30, loại bằng chứng visual/OCR/ASR và độ khó.
- Người viết query chỉ xem raw clip, không xem OCR, ASR, ranking hoặc output hệ thống.
- Hai annotator độc lập xác định raw interval rồi adjudicate.
- KIS/QA dùng `[s,e]` thật.
- QA có answer aliases và semantic-judge protocol.
- TRAKE có đúng video, thứ tự và interval riêng cho từng event.
- Test GT không để người tuning truy cập.

### 9.3 Thống kê

- Mọi so sánh paired theo cùng query.
- Bootstrap 10.000 lần, cluster theo video.
- Seed lấy từ SHA-256, không dùng Python `hash()`.
- Báo `Delta Final`, CI 95% và win/tie/loss.
- McNemar cho hit/miss; paired permutation cho Final.
- Grid validation dùng nested CV hoặc multiple-comparison correction.
- Primary metric phải đăng ký trước; subgroup chỉ dùng chẩn đoán.

## 10. Gate promote hoặc dừng

### 10.1 Candidate gate trên validation

Policy đi tiếp nếu thỏa ít nhất một điều:

- correct-video hoặc true-interval recall tăng ít nhất 3 điểm phần trăm; hoặc
- Oracle Final tăng ít nhất `0.010`.

Đồng thời:

- không task/modality đủ lớn nào giảm quá 2 điểm phần trăm;
- không vượt `B_pool` và latency budget;
- pool luôn deterministic và đủ đúng số candidate sau refill.

### 10.2 End-to-end gate trên locked test

Thay `submission_v2` chỉ khi thỏa đồng thời:

- `Delta Final >= +0.010`;
- cận dưới CI 95% không âm;
- candidate oracle không giảm;
- `R@100` không giảm quá `0.005`;
- không task có ít nhất 30 query giảm quá 3 điểm phần trăm;
- P95 latency và chi phí không tăng quá 5% ở cùng ba ngân sách.

Ngoại lệ non-inferiority: có thể nhận cấu hình rẻ hơn ít nhất 20% nếu
`Delta Final >= -0.005`.

Nếu union/hybrid tăng oracle nhưng Final không tăng, bottleneck là VLM selector hoặc
final ranker. Không tăng tiếp `B_pool` và không kết luận admission vô ích.

## 11. Artifact và khả năng tái lập

Mỗi run phải nằm ở thư mục riêng:

```text
artifacts/runs/<run_id>/
  manifest.json
  candidate_trace.jsonl.zst
  source_topk.npz
  vlm_scores.jsonl.zst
  submissions/
  per_query_scores.jsonl
  summary.json
  timings.json
  comparison.json
```

`manifest.json` tối thiểu chứa:

- dataset/split/GT hash;
- Git SHA và dirty state;
- model ID, revision và remote-code SHA;
- prompt SHA;
- index và source-data hashes;
- candidate/VLM/final-ranking policy;
- `B_pool`, quota từng nguồn, `K_vlm`, `M_emit`, spread và output budget;
- deterministic seed;
- command đầy đủ;
- start/end time, GPU-seconds và chi phí;
- output file hashes.

Mọi query thiếu, duplicate query ID, thiếu budget hoặc malformed CSV phải làm run fail;
không được bỏ khỏi mẫu số.

## 12. Roadmap tối ưu toàn hệ

### P0 — Correctness và reproducibility

Thời gian dự kiến: 3–5 ngày kỹ thuật; annotation chạy song song.

- [ ] Track scorer, `src/scoring`, tests, GT schema và preprocessing configs.
- [ ] Thêm CI chạy từ clean clone.
- [ ] Scorer task-aware cho KIS, QA và TRAKE.
- [ ] Missing/extra query hard fail.
- [ ] Stable SHA seed và paired jitter samples.
- [ ] Nối `run.py` qua writer/validator.
- [ ] Output temp-to-atomic-rename; không để stale CSV.
- [ ] Pin model revision và xác minh manifest lúc chạy.
- [ ] Version query cache theo model/dim/preprocess/prompt.
- [ ] Lưu run manifest, timing và cost.
- [ ] Hoàn thành independent validation và locked test GT.

Gate P0:

- clean clone chạy được scorer;
- ba lần replay sinh cùng output hash;
- 100% output qua validator;
- query thiếu làm run fail;
- model/index/cache mismatch bị chặn trước inference.

### P1 — Candidate recall và reranker

Thời gian dự kiến: 4–6 ngày; GPU cap đề xuất `$10`.

- [ ] Candidate policy interface F/U/H.
- [ ] Positive-hit eligibility cho lexical source.
- [ ] Global budget, dedup và deterministic refill.
- [ ] Source/fused top-1000 trace.
- [ ] Candidate oracle sweep.
- [ ] VLM superset cache.
- [ ] VLM-F/U/H sweep.
- [ ] Final ranker small-grid/nested-CV.
- [ ] One-shot locked test.

Deliverable:

```text
artifacts/eval/topk_v1/
  pareto.json
  pareto.html
  candidate_policy.yaml
  vlm_policy.yaml
  validation_report.json
  locked_test_report.json
```

### P2 — Temporal refinement

Thời gian dự kiến: 3–5 ngày.

Thay spread hình học bằng coarse-to-fine raw-frame refinement:

1. Lấy top shot/moment từ candidate policy đã khóa.
2. Decode frame thật trong shot hoặc local window.
3. Coarse stride 4–8 bằng image encoder.
4. Giữ local peak.
5. Fine stride 1 trong khoảng khoảng 8–16 frame quanh peak.
6. Optional VLM trên một số raw frame tốt nhất.
7. Chỉ phát frame đã decode và score; spread giữ làm fallback.

TRAKE phải refine từng probe rồi chạy DP trên timestamp milliseconds, không dùng raw frame
distance với lambda được hiểu theo giây.

Gate:

- true-interval hit@100 tăng ít nhất 5 điểm phần trăm;
- Final tăng ít nhất `0.010`;
- wall time tăng không quá 50% hoặc vẫn nằm trong deadline tier.

### P2 — QA

Thời gian dự kiến: 2–3 ngày sau khi có QA GT.

- [ ] Lấy top 5–10 candidate khác video/shot.
- [ ] Sinh answer riêng cho từng candidate.
- [ ] Ghép OCR/ASR cục bộ đúng candidate.
- [ ] Lưu answer confidence và evidence.
- [ ] Joint rank bằng retrieval score và answer confidence đã calibrate.
- [ ] Normalize aliases và semantic judge.

Đo ba tầng riêng:

- retrieval hit;
- reader accuracy khi đưa GT frame;
- joint QA score.

Gate: reader-on-GT ít nhất 70% trước khi bật reader đó cho production.

### P2 — TRAKE

Thời gian dự kiến: 3–5 ngày sau khi có ít nhất 35–50 query.

- [ ] Candidate/event coverage report.
- [ ] VLM hoặc local rerank theo từng event/path.
- [ ] DP dùng milliseconds.
- [ ] Grid lambda/min-gap/k-path chỉ trên train/validation.
- [ ] Test N=1 vẫn tương đương KIS.

Chỉ đổi `lambda=0` nếu `Delta Final >= +0.020` và cận dưới CI 95% dương.

### P3 — Performance và operational hardening

Thời gian dự kiến: 4–6 ngày; ít nhất ba full dry-run.

- [ ] `modal.Cls`/`@enter` preload Jina và Qwen.
- [ ] Batch liên query và retry idempotent.
- [ ] Persist BM25 index.
- [ ] Gộp object JSON thành Parquet/SQLite/mmap.
- [ ] Không đọc OCR/ASR lần hai cho QA.
- [ ] Cache atomic, revisioned và có lock.
- [ ] Resumable stage theo run ID.
- [ ] Deadline tier: retrieval → VLM → temporal refinement.
- [ ] Failure injection: mất batch, retry và resume.

Gate:

- cold source load từ khoảng 40 giây xuống dưới 10–15 giây;
- 100 query end-to-end dưới 30 phút;
- ít nhất 3 lần headroom so với deadline 2 giờ 30;
- ba dry-run không thiếu file, duplicate hoặc stale output;
- resume không làm thay đổi output hash.

### P4 — Tối ưu lựa chọn model

Chỉ thực hiện sau khi P1–P3 ổn định để không đánh giá model trên pipeline đang đổi.

- [ ] Jina 512 so với supplied CLIP, SigLIP/OpenCLIP hoặc challenger phù hợp.
- [ ] OCR có human GT và CER/WER.
- [ ] ASR có human transcript và WER theo source/video group.
- [ ] Object/crop chỉ bật theo query routing nếu thắng validation độc lập.
- [ ] Qwen/VLM challenger dùng cùng cached candidate set và prompt protocol.

Model chỉ được thay khi thắng end-to-end trên locked test, không chỉ thắng một proxy metric.

## 13. Trình tự phụ thuộc

```text
P0 Correctness + reproducibility
            │
            ▼
P1 Candidate admission F/U/H
            │
            ▼
P1 VLM shortlist F/U/H
            │
            ▼
P1 Final ranking
            │
            ▼
P2 Raw-frame temporal refinement
            │
            ├── QA candidate-specific reader
            └── TRAKE per-event/path rerank
            │
            ▼
P3 Performance hardening
            │
            ▼
P4 Model replacement
```

Không tối ưu model, spread, fusion weight, candidate policy và VLM weight trong cùng một
vòng. Mỗi vòng chỉ thay một lớp, lưu trace đủ để xác định lợi ích đến từ đâu.

## 14. Timeline gợi ý

Với một kỹ sư chính và hỗ trợ annotation:

| Tuần | Kết quả phải có |
|---|---|
| 1 | P0 scorer/validator/provenance; frozen baseline; trace schema |
| 2 | Candidate F/U/H sweep; Pareto frontier; VLM superset cache |
| 3 | VLM selector/final ranking; validation report; freeze challenger |
| 4 | Locked test; raw-frame refinement pilot |
| 5 | QA và TRAKE pipeline/eval |
| 6 | Performance hardening, failure test và full dry-run |
| 7 dự phòng | Model challenger hoặc sửa failure còn lại |

Compute cap đề xuất cho toàn roadmap trước model replacement: khoảng `$40`, không tính
công gán nhãn. Phần lớn sweep P1 phải chạy offline từ score/VLM cache.

## 15. Definition of Done

Hệ thống chỉ được coi là tối ưu và sẵn sàng thi khi:

- [ ] Clean clone dựng được evaluator và chạy CI xanh.
- [ ] Model/index/cache mismatch bị chặn.
- [ ] Có independent validation và locked test với raw interval thật.
- [ ] Candidate policy thắng theo gate đã đăng ký trước.
- [ ] Mọi VLM query-frame được cache/provenance đầy đủ.
- [ ] QA trả answer riêng từng candidate.
- [ ] TRAKE được eval trên ít nhất 35–50 query official-like.
- [ ] Frame phát ra đã được decode/score hoặc được đánh dấu rõ là fallback spread.
- [ ] Output atomic, đủ query, đúng budget và qua validator.
- [ ] Ba full dry-run thành công trong time/cost cap.
- [ ] Mỗi score công bố có dataset hash, Git SHA, config và output hash.

## 16. Sprint đầu tiên cần làm ngay

Thứ tự triển khai cụ thể:

1. Khóa và hash `submission_v2` làm baseline.
2. Sửa evaluator deterministic và missing-query hard fail.
3. Tạo `CandidatePolicy` cùng trace top-1000/source + fused.
4. Sửa lexical eligibility thành `covered & score > 0`.
5. Thêm global dedup/refill/budget.
6. Chạy candidate oracle sweep F/U/H trên dev100.
7. Chọn tối đa bốn Pareto candidates.
8. Chấm một VLM superset duy nhất.
9. Replay VLM-F/U/H và dựng funnel report.
10. Chưa thay production config cho tới khi có independent validation.
