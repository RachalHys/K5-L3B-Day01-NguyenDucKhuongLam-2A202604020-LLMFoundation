# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED:** 2018.

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — bài làm, có output Phần 0–3, 4a và bảng so sánh | ✓ |
| [`lab1_report.json`](lab1_report.json) — 15 kiểm tra đạt, 10 dự đoán và 6 phương án | ✓ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh năm phép tính tay có các bước | ✓ |


## 2. Kết quả chính

Dữ liệu sau bỏ trùng có 5.169 tin; train/val/test gồm 3.619/775/775 tin. Tập test có 94 spam; SUB gồm 20 spam và 20 ham.

| Phương án | precision | recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 1.000 | 0.840 | 0.913 | 0.919 | 9.221 | 19287.180 | 0.089 | 1.151 | 0.000 |
| 2. BoW + MLP (deep learning) | 1.000 | 0.904 | 0.950 | 0.974 | 9.926 | 7223.758 | 0.265 | 8.587 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.846 | 0.936 | 0.889 | 0.950 | 157.792 | 18.088 | 44.472 | 66.736 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.945 | 0.915 | 0.930 | 0.974 | 158.453 | 29.870 | 27.806 | 41.265 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.924 | 0.904 | 0.914 | 0.974 | 158.858 | 42.330 | 47.307 | 76.325 | 0.000 |
| 4a. LLM nhỏ zero-shot | 1.000 | 0.950 | — | 0.974 | 0.000 | 0.090 | 1426.031 | 31386.587 | 0.027 |

Gemini 3.5 Flash-Lite zero-shot đã hoàn thành trên SUB 40 tin, F1 = 0.974 và invalid = 1. Một tin spam bị bộ lọc nội dung chặn, trả phản hồi rỗng và được gán ham theo quy tắc xử lý của bài. Precision và recall của LLM cũng đo trên SUB; không có F1 trên toàn bộ test. Khi so sánh với LLM, dùng F1 trên cùng SUB 40 tin. Chênh lệch F1 dưới khoảng 0.05 trên tập nhỏ này chưa đủ để xếp hạng chắc chắn.

Chi phí ước lượng là 0.00107 USD/40 tin, tương đương khoảng 0.027 USD/1.000 tin, chưa tính token suy luận. Tổng thời gian lô gồm khoảng nghỉ 5 giây giữa các request; p50/p95 không gồm khoảng nghỉ chủ động này. Lượt 4b đã bị ngắt; chưa có kết quả 4b, 4c và stress test. JSON lưu 15/15 kiểm tra đạt và đủ 10 dự đoán, nhưng các kiểm tra này không thay thế kết quả thực nghiệm còn thiếu.

## 3. Dự đoán và thực tế

| Thí nghiệm | Dự đoán | Kết quả và giải thích |
| --- | --- | --- |
| Learning rate | lr=50 dao động, không hội tụ | Loss dao động mạnh ban đầu rồi giảm về 0.005. Dự đoán chỉ đúng ở đoạn đầu; bước học lớn vẫn làm giảm loss train trên dữ liệu này. |
| MLP | F1 kém LogReg vì dễ overfit | F1 MLP 0.950, cao hơn LogReg 0.913 khoảng 0.036. Dù val loss tăng sau chỉ số epoch 6, khả năng phân loại trên test vẫn tốt hơn LogReg. |
| Bỏ ReLU | F1 giảm về gần LogReg | F1 đạt 0.944, gần MLP có ReLU 0.950 hơn LogReg 0.913. Hai lớp Linear tương đương một lớp Linear nhưng cách tối ưu và số bước train khác nhau nên F1 không nhất thiết bằng nhau. |

## 4. Phân tích 50 case khó

Chưa có kết quả chạy bộ 50 case khó. Ô thêm case dừng vì chưa nạp `HARD_EVAL`, nên chưa có dự đoán cho năm case bổ sung: ham chứa từ khóa spam, quảng cáo việc làm, mạo danh người thân, né ký tự và tin đang nói về prompt injection.

## 5. Thử thách cá nhân

Đề theo SEED 2018: so sánh MLP có 4 và 512 nơ-ron ẩn, cùng seed, 30 epoch, batch size 64 và Adam với lr=0.001.

| Nơ-ron ẩn | Số tham số | F1 test | Train (s) |
| ---: | ---: | --- | --- |
| 4 | 8009 | — | — |
| 512 | 1025025 | — | — |

Mạng nhỏ phù hợp khi chất lượng chấp nhận được và cần tiết kiệm tài nguyên. Mạng lớn có khả năng biểu diễn mạnh hơn nhưng cũng dễ overfit; chưa có kết quả để chọn giữa hai cấu hình.

## 6. Phương án cho sản phẩm thật

Tạm chọn BoW + MLP: F1 test 0.950, p95 8.587 ms và thông lượng cả lô khoảng 7.224 tin/giây. Trong phép đo hiện tại, mô hình đạt p95 dưới 50 ms và vượt mức trung bình khoảng 116 tin/giây của 10 triệu SMS/ngày. So với LogReg, p95 cao hơn nhưng số spam bỏ sót giảm từ 15 xuống 9 trên 94 tin; cả hai không chặn nhầm ham trên tập test.

BoW + LogReg, BoW + MLP và MiniLM + LogReg đạt p95 dưới 50 ms. Đường Pareto theo F1 SUB và p95 gồm BoW + LogReg và BoW + MLP. Các số đo này chưa gồm tải đồng thời và đỉnh lưu lượng. Có thể dùng LogReg xử lý tin có xác suất rõ ràng, rồi chuyển tin lưỡng lự sang embedding hoặc LLM. LLM 4a có F1 SUB 0.974 nhưng p95 khoảng 31,39 giây, vượt yêu cầu 50 ms. Từ chi phí ước lượng 0.02682 USD/1.000 tin trong JSON, xử lý toàn bộ 10 triệu tin/ngày trong 30 ngày sẽ tốn khoảng 8.046 USD theo giá trả phí, nếu độ dài prompt và phản hồi tương tự; con số này chưa tính token suy luận.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 1–3 | So sánh BoW và MLP trong phân loại SMS. MLP nhiều tham số có tốt hơn không? Rà soát gradient, embedding và các lỗi trong bài làm. | Giải thích cơ chế, hỗ trợ chỉnh code, dự đoán và câu trả lời; sửa điều kiện tính recall. | Đối chiếu công thức và shape với đề, kết quả Phần 1–3 với output Colab và JSON. Các ô kiểm tra tương ứng đều đạt; MLP có F1 0.950, cao hơn LogReg 0.913. |
| Phần 4 | Dùng Gemini qua Google AI Studio theo template. Tạo prompt, đọc nhãn, đo latency, tính chi phí và giới hạn tốc độ gọi theo quota. | Hỗ trợ code phân loại theo lô, nghỉ 5 giây trước mỗi request và chờ tăng dần khi retry. | Lượt 4a đã hoàn thành trên 40 tin, F1 = 0.974, invalid = 1 do bộ lọc nội dung. Lượt 4b bị ngắt, chưa có kết quả 4c và stress test LLM; đã gặp giới hạn API 15 request/phút. Tổng thời gian lô gồm khoảng nghỉ 5 giây, p50/p95 không gồm khoảng nghỉ chủ động này. |
| Thử thách và case bổ sung | Hoàn thiện phần so sánh MLP 4/512 nơ-ron và các case khó có giải thích bẫy. | Hỗ trợ code so sánh và thêm năm case kèm why_tricky. | Số tham số khớp công thức. Chưa chạy thử thách trên dữ liệu SMS nên chưa có F1 và thời gian train; ô thêm case dừng vì chưa nạp HARD_EVAL nên chưa có kết quả năm case. |
| Tính tay và báo cáo | Trình bày năm phép tính, tổng hợp kết quả và báo cáo theo mẫu. | Hỗ trợ tính toán, đối chiếu số liệu và chỉnh câu chữ. | Năm giá trị làm tròn khớp hash của đề; đã có ảnh năm phép tính tay. Bảng sáu phương án khớp JSON và output Colab; F1 test của LLM để trống vì chỉ chạy SUB. Các kết quả chưa chạy không được điền. |

MLP cải thiện recall trên BoW từ 0.840 lên 0.904, dù số tham số lớn và có dấu hiệu overfit. MiniLM + LogReg đúng 5/6 tin trong thí nghiệm 3.B, cao hơn BoW với 4/6 tin, nhưng độ trễ lớn hơn; biểu diễn tốt hơn không tự động đồng nghĩa là phương án phù hợp hơn cho mọi yêu cầu.
