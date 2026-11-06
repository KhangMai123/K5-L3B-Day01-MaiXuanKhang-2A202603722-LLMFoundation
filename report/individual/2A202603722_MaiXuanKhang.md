# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **Thông tin định danh:** thể hiện trong tên tệp báo cáo theo quy định của lab.
- **SEED:** 3491.
- **Thử thách được giao:** #6, vì `3491 % 6 = 5` (chỉ số bắt đầu từ 0).
- **LLM sử dụng:** DeepSeek Flash, model API `deepseek-flash`, `thinking.type="disabled"`.

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook có 100 cell, 65 cell mã đã chạy và còn output | Có |
| [`lab1_report.json`](lab1_report.json) — báo cáo do cell cuối notebook sinh ra | Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh viết tay đầy đủ 5 phép tính và các bước | Có |

Notebook và JSON cùng SEED 3491, có đủ 10 dự đoán và 15/15 kiểm tra đạt. Notebook không có output exception. Các mốc dự đoán được ghi trước hoặc cùng giây với lời gọi `require_prediction`; độ phân giải log chỉ đến giây nên không khẳng định thứ tự chi tiết hơn. API key được để trống trong bản nộp; khi chạy lại cần điền key hoặc dùng Secret `DEEPSEEK_API_KEY`.

Năm đáp án tính tay khớp ảnh và hash kiểm tra: `baseline_accuracy=0.87` (tỷ lệ chưa làm tròn tương đương 87.37%), `sigmoid_0=0.50`, `f1_example=0.73`, `cosine_example=0.60`, `llm_cost_per_day=10.50`. Bài chi phí viết tay dùng đúng giả thiết GPT-6 Luna trong đề, tách biệt bảng giá DeepSeek của lần chạy API.

## 2. Kết quả chính

| Phương pháp | F1 test | F1 SUB 40 | Train (s) | Tin/s cả lô | p95 (ms/tin) | USD/1k tin (ước tính) |
| --- | --- | --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.9130 | 0.8889 | 6.47 | 29782.77 | 0.239 | 0.00000 |
| 2. BoW + MLP (deep learning) | 0.9474 | 0.9744 | 13.53 | 9774.43 | 4.447 | 0.00000 |
| 3a. Encoder (MiniLM) + kNN | 0.9143 | 0.9744 | 84.93 | 24.96 | 120.441 | 0.00000 |
| 3b. Encoder (MiniLM) + LogReg | 0.9436 | 0.9744 | 85.40 | 36.70 | 34.306 | 0.00000 |
| 3c. Encoder (MiniLM) + MLP head | 0.9534 | 0.9744 | 86.13 | 35.56 | 31.432 | 0.00000 |
| 4a. LLM nhỏ zero-shot | — | 1.0000 | 0.00 | 1.06 | 1155.969 | 0.02343 |
| 4b. LLM lớn zero-shot | — | 1.0000 | 0.00 | 0.99 | 1174.745 | 0.02343 |
| 4c. LLM nhỏ few-shot | — | 1.0000 | 0.00 | 1.11 | 1191.816 | 0.05903 |

Train/val/test lần lượt 3619/775/775 tin; test có 100 spam. SUB là 40 tin cân bằng (20 spam, 20 ham), lấy từ test. Hàng LLM chỉ đánh giá trên SUB, nên không coi F1=1.000 của LLM là F1 toàn bộ 775 tin. Giá và token trong notebook là ước tính từ số từ/0.75; không phải `usage` thực hay hóa đơn, chưa tính đầy đủ cache-hit/off-peak.

Hai hàng 4a và 4b đều gọi cùng `deepseek-flash`; đây là hai lượt chạy cùng model, không phải bằng chứng model lớn hơn model nhỏ. Cả hai đúng 40/40, invalid=0. Few-shot cũng đúng 40/40, phí ước tính tăng 2.52 lần nhưng chưa cải thiện F1 trên tập nhỏ này. Một lỗi trên SUB có thể làm F1 đổi khoảng 2.5 điểm %, nên không xếp hạng chắc chắn từ chênh lệch một tin.

Trên toàn test, Embedding+MLP head có F1 cao nhất trong các model local: 0.9534. Pareto của biểu đồ SUB gồm BoW+LogReg, BoW+MLP và LLM zero-shot 4a. LLM đạt chất lượng tốt nhất trên SUB nhưng có p95 hơn 1 giây; nằm trên Pareto không đồng nghĩa đạt SLA 50ms.

## 3. Dự đoán và thực tế

Ba đối chiếu dưới đây có độ lệch hoặc điều chỉnh kỳ vọng rõ nhất. Nhiều dự đoán gốc dùng từ “có thể”, nên không coi chúng là khẳng định định lượng chắc chắn.

1. **`hard_eval`:** dự đoán accuracy bộ khó khoảng 50–80%. Các model local nằm trong khoảng này (52–72%), nhưng DeepSeek zero-shot đạt 98% và prompt phòng thủ đạt 96%, cao hơn kỳ vọng. LLM xử lý được nhiều ngữ cảnh/từ bị biến dạng mà BoW hoặc head embedding học từ dữ liệu SMS cũ chưa xử lý tốt; kết quả vẫn chỉ áp dụng bộ 50 case được chọn có chủ đích.
2. **`llm_fewshot`:** kỳ vọng bốn ví dụ có thể tăng F1. Thực tế zero-shot và few-shot cùng F1=1.000 trên SUB; chi phí ước tính tăng từ 0.02343 lên 0.05903 USD/1k tin. Baseline đã đúng hết 40 tin nên bộ này không còn chỗ để thể hiện cải thiện; ví dụ làm dài input mà chưa đem lại lợi ích chất lượng quan sát được.
3. **`ml_f1`:** dự đoán F1 khoảng 0.90, precision cao hơn recall. Thực tế F1=0.9130 (cao hơn khoảng 1.30 điểm F1), precision=1.00, recall=0.84. Dự đoán chiều precision/recall đúng; độ lệch nhỏ cho thấy các từ như call/txt/free đã tạo baseline hữu ích, đồng thời mô hình thận trọng nên bỏ lọt 16/100 spam. Không gọi đây là một sai lệch lớn hơn dữ liệu thực tế cho phép.

## 4. Phân tích 50 case khó

| Phương pháp | F1 SUB 40 | F1 bộ khó | Đúng trên 50 case |
| --- | --- | --- | --- |
| BoW + LogReg | 0.8889 | 0.3333 | 26/50 (52%) |
| BoW + MLP | 0.9744 | 0.4615 | 29/50 (58%) |
| Embedding + kNN | 0.9744 | 0.7586 | 36/50 (72%) |
| Embedding + LogReg | 0.9744 | 0.6923 | 34/50 (68%) |
| Embedding + MLP head | 0.9744 | 0.6531 | 33/50 (66%) |
| DeepSeek zero-shot | 1.0000 | 0.9825 | 49/50 (98%) |
| DeepSeek + prompt phòng thủ | chưa đo riêng | 0.9643 | 48/50 (96%) |

BoW+LogReg tụt nhiều nhất so với SUB: 0.8889→0.3333, giảm khoảng 55.56 điểm F1. BoW+MLP đặc biệt yếu ở né từ khóa (0% đúng); BoW+LogReg yếu ở spam giọng tự nhiên (10%). Embedding+kNN chỉ đúng 25% ở nhóm nói về spam, cho thấy gần ngữ nghĩa với nội dung spam không đồng nghĩa người gửi đang phát tán spam.

DeepSeek zero-shot và prompt phòng thủ cùng đúng 6/6 ở nhóm prompt injection. Phòng thủ không cải thiện nhóm nào trong lần chạy này; nhóm spam giọng tự nhiên giảm từ 90% xuống 80%. Tên “F1 test thường” của hàng 4d trong code gốc mượn giá trị 4a; chưa có phép đo SUB riêng cho prompt phòng thủ, vì vậy bảng báo cáo ghi “chưa đo riêng”.

Hai case cụ thể:

- **#16 — `natural_spam`:** tin bạn cũ giới thiệu việc kinh doanh skincare và xin gửi catalogue. Tất cả model đều sai, gồm cả hai prompt LLM. Văn bản giống một cuộc trò chuyện cá nhân và không tự chứng minh là liên hệ không mong muốn; thông tin người gửi/đồng ý nhận tin bị thiếu. Đây cũng là giới hạn của nhãn viết tay.
- **#13 — `natural_spam`:** tin thông báo ảnh sự kiện đã sẵn sàng, kèm đường dẫn xem/đặt in. Zero-shot đúng nhưng prompt phòng thủ đoán ham, tạo thêm một FN. Model có thể nhận đây là thông báo giao dịch hợp lệ; thay đổi prompt làm thay đổi ranh giới nhãn và không tự động tăng độ an toàn.

Các case cùng nguồn/đúng mẫu trong test thường không đại diện hết spam mới. Bộ khó hữu ích để tìm failure mode, kiểm tra hồi quy và xây dựng giả thuyết; không đủ để ước lượng chính xác tỷ lệ lỗi vận hành.

**Case bổ sung:** `MY_CASES` có 5 tin về giả người thân chuyển tiền, thanh toán thuê bao, phí giao hàng, đầu tư và quảng cáo. Output đã lưu cho thấy chúng được thêm sau phép đánh giá 50 case; chưa có kết quả chạy lại trên 55 case hoặc bằng chứng chúng đánh lừa DeepSeek. Các case có AI hỗ trợ xây dựng, không tự nhận là phần bonus hoàn toàn tự nghĩ ra.

## 5. Thử thách cá nhân

**Đề #6:** cho LLM trả JSON `{"label": ..., "reason": ...}`, parse và so với một từ trên 10 tin SUB.

Đã viết `build_json_prompt` và `parse_json_label`. Parser dùng `json.loads`, kiểm tra đúng hai key, label thuộc `spam/ham`, reason là chuỗi. Code gọi cả hai dạng trên cùng 10 tin (5 spam, 5 ham), lưu dự đoán, token/chi phí ước tính và latency.

| Chỉ số trên 10 tin | Một từ | JSON + reason |
| --- | ---: | ---: |
| Số tin đúng | 10/10 | 10/10 |
| Accuracy / F1 | 1.000 / 1.000 | 1.000 / 1.000 |
| Tổng token ước tính | 752.00 | 802.67 |
| Tổng USD ước tính | 0.0002376 | 0.0003944 |
| Tổng latency (s, xấp xỉ từ output) | 8.184 | 10.045 |

JSON tăng tổng token khoảng 6.74% và phí khoảng 1.66 lần vì có nhiều output token hơn, trong khi output có đơn giá cao hơn input. Prompt hai dạng cũng khác nhau về độ dài nên không gán toàn bộ chênh lệch token cho riêng trường reason. Trên 10 tin này JSON chưa tăng độ đúng; lý do hữu ích cho review, còn tuyến lọc lớn chỉ cần nhãn thì cân nhắc chi phí/độ trễ bổ sung. Phần code và lời giải thử thách có AI hỗ trợ; không tự nhận đã làm hoàn toàn độc lập trong 15 phút.

## 6. Phương án cho sản phẩm thật

10 triệu SMS/ngày tương đương khoảng 115.74 tin/s trung bình; cần chừa công suất cho giờ cao điểm và hàng đợi.

- **Tuyến nhanh:** BoW+MLP có p95=4.447ms, khoảng 9774 tin/s cả lô, F1 test=0.9474; phù hợp làm baseline xử lý nhanh. BoW+LogReg còn nhanh hơn nhưng recall spam chỉ 0.84.
- **Nhánh tăng chất lượng:** Embedding+MLP có F1 test=0.9534 và p95=31.432ms, nhưng throughput chỉ 35.56 tin/s ở cấu hình đo. Cần tăng worker/tài nguyên hoặc chỉ nhận một phần traffic. Embedding+LogReg tương tự: p95=34.306ms, khoảng 36.70 tin/s.
- **Không dùng trực tiếp cho SLA hiện tại:** kNN có p95=120.441ms; LLM API có p95 khoảng 1156–1192ms và chỉ khoảng 1 tin/s khi gọi tuần tự. Thêm song song có thể tăng throughput, không tự khắc phục p95 từng request.

Đề xuất router: local MLP xử lý tin có xác suất đã hiệu chỉnh rất thấp/cao; vùng chưa chắc chắn, ca bất đồng và sender/domain rủi ro chuyển sang embedding hoặc LLM/review. Ngưỡng `[0.3,0.7]` chỉ là khởi đầu để tune trên val. Với OTP cần ràng buộc FP riêng. Nếu LLM được gọi đồng bộ, phải đo lại p95 toàn hệ thống; xử lý bổ sung bất đồng bộ là một lựa chọn để giữ tuyến nhanh.

Ngoại suy phí từ prompt đã đo: zero-shot khoảng **7029 USD/30 ngày** nếu gọi cho toàn bộ 10 triệu tin/ngày; few-shot khoảng **17709 USD/30 ngày**. Nếu chỉ 5% tin gọi zero-shot, phí token ước tính khoảng **351.45 USD/tháng**, chưa tính máy local. Tỷ lệ 5% là giả định thiết kế, chưa được đo. Các con số dựa trên công thức ước tính của notebook, không phải báo giá hay hóa đơn API thực.

## 7. AI log và điều đã học

| Phạm vi | Hỏi AI gì | Nhận gì | Kiểm chứng / giới hạn |
| --- | --- | --- | --- |
| Phần 1–3 | Hoàn thiện TODO và giải thích NumPy/PyTorch/embedding | AI hỗ trợ viết các hàm và lời giải | Có output kiểm tra và kết quả train/test; không tự nhận toàn bộ code là tự viết |
| Phần 4 | Chuyển Gemini sang DeepSeek Flash, sửa cấu hình gọi API | Hỗ trợ cấu hình model/endpoint/key và tắt thinking | Output cho thấy DeepSeek chạy thật; 4a/4b cùng model; key xóa khỏi bản commit |
| Câu hỏi và tính tay | Yêu cầu giải thích công thức, đối chiếu các thí nghiệm | Lời giải và năm công thức | Ảnh tính tay có các bước; 5 hash đều đạt |
| Thử thách #6 / case bổ sung | Viết prompt/parser JSON và xây dựng ca khó | AI hỗ trợ code và case | JSON đã chạy 10 tin; chưa có output LLM trên 5 case mới |
| Chuẩn bị nộp bài | Rà soát ba tệp, điền báo cáo và commit | Báo cáo theo số liệu đã lưu, cập nhật các câu trả lời cũ, bỏ API key | Không gọi thêm API; giữ kết quả chạy thật và phân biệt phần chưa có bằng chứng |

Điều học được: biểu diễn pretrained và điểm test cao không bảo đảm bền trước spam khác phân phối; cần đọc từng lỗi. Model chất lượng cao có thể không đạt SLA, còn thêm few-shot hay prompt phòng thủ phải được kiểm nghiệm vì lần này không tăng F1 SUB và phòng thủ còn làm sai thêm một case. Kết quả nhỏ cần đọc cùng confusion matrix, dữ liệu và giới hạn phép đo.
