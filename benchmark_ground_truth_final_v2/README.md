# AIC2026 Final Ground Truth

Gói này chứa Ground Truth shot-level đã hoàn tất để chạy benchmark retrieval/fusion.
Dataset frame/video không được đóng gói vì dung lượng lớn.

## Files

- `ground_truth_final.jsonl`: file benchmark chính, một JSON object trên mỗi dòng.
- `ground_truth_final.csv`: bản CSV tương đương để kiểm tra hoặc phân tích.
- `ground_truth_report.json`: thống kê và kết quả validation.
- `excluded_queries.jsonl`: query bị loại; file rỗng nghĩa là không có query bị loại.

## Fields chính

- `query_id`, `query_text`, `query_type`: định danh và nội dung truy vấn.
- `positive_shots`: shot Ground Truth relevance 2; đây là đơn vị chấm chính.
- `partial_shots`: shot liên quan một phần, relevance 1.
- `hard_negative_shots`: shot rất giống nhưng sai chi tiết bắt buộc.
- `representative_frame`: frame đại diện thuộc positive shot, không phải đơn vị chấm chính.
- `evidence`: Vision/OCR/ASR evidence gốc của query.
- `status`: tất cả record trong file chính phải là `APPROVED`.

## Đọc JSONL bằng Python

```python
import json
from pathlib import Path

path = Path("ground_truth_final.jsonl")
queries = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

for item in queries:
    query_text = item["query_text"]
    ground_truth_shots = {
        shot["shot_id"] for shot in item["positive_shots"]
    }
```

## Đọc CSV bằng Python

```python
import csv

with open("ground_truth_final.csv", encoding="utf-8", newline="") as file:
    rows = list(csv.DictReader(file))
```

## Dùng để test retrieval/fusion

Với mỗi query, đưa `query_text` vào pipeline retrieval/fusion, chuyển kết quả về
`shot_id`, rồi so sánh ranking với tập `positive_shots[*].shot_id`. Không chấm
theo exact frame. Có thể dùng `partial_shots` cho graded metrics như nDCG và dùng
`hard_negative_shots` để phân tích lỗi.

Không hard-code đường dẫn dataset từ máy tạo benchmark. Máy chạy test cần tự cấu
hình dataset/index bằng đường dẫn local của máy đó; các `shot_id` và frame key
trong GT là khóa tương đối để mapping vào corpus.
