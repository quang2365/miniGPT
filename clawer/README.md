# DPO dataset 490

dataset này dùng model nhỏ `qwen2.5:7b` để dịch một bộ dữ liệu DPO từ tiếng Anh sang tiếng Việt, phục vụ cho việc huấn luyện mô hình preference tuning hoặc fine-tuning theo hướng alignment.

## Mục tiêu

- Dịch bộ dữ liệu UltraFeedback binarized preferences từ gốc tiếng Anh sang tiếng Việt.
- Giữ nguyên cấu trúc dữ liệu gốc: `instruction`, `chosen_response`, `rejected_response`.
- Tạo dataset đã dịch theo định dạng JSON để dễ dùng cho quá trình SFT/DPO tiếp theo.

## Nguồn dữ liệu

- Dataset gốc: `https://huggingface.co/buckets/quang2365/ultrafeedback-binarized-preferences-bucket`
## Script chính

- File xử lý: `./translate.py`
- Script dùng `AsyncOpenAI` để gọi mô hình qua endpoint Ollama local:
  - `http://localhost:11434/v1`
  - model: `qwen2.5:7b`

Các bước chính trong script:

1. Tải dataset parquet bằng `datasets.load_dataset`.
2. Lấy các cột:
   - `instruction`
   - `chosen_response`
   - `rejected_response`
3. Với mỗi sample, tạo prompt yêu cầu mô hình dịch toàn bộ triplet sang tiếng Việt.
4. Dùng `response_format` để ép output theo schema:
   - `prompt`
   - `chosen`
   - `rejected`
5. Lưu từng dòng dịch sang file JSON theo dạng list.

## Định dạng dữ liệu gốc

Dataset gốc có cấu trúc tương tự:

```json
"train":{
  "instruction": "Given a question, produce a helpful and safe answer.",
  "chosen_response": "...",
  "rejected_response": "...",
  "...":"..."
}
```

Trong đó:
- `instruction`: câu lệnh / yêu cầu đầu vào.
- `chosen_response`: câu trả lời tốt hơn (được chọn).
- `rejected_response`: câu trả lời kém hơn (bị từ chối).

## Định dạng sau khi dịch

Script sẽ chuyển về cấu trúc:

```json
[
  {
    "prompt": "Bạn có thể ...",
    "chosen": "Câu trả lời tốt hơn đã được dịch sang tiếng Việt ...",
    "rejected": "Câu trả lời kém hơn đã được dịch sang tiếng Việt ..."
  }
]
```

- Do hạn chế về model, phân cứng và thời gian nên dataset chỉ dịch 490 mẫu so với 60000 mẫu ban đầu, bạn có thể tuy chọn model thích hợp với phần cứng và chạy thêm
