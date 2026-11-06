# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED:** 2018.
- **Phạm vi thực hiện:** Phần 0–3; 8/8 kiểm tra đã chạy đạt, ghi nhận 6/10 dự đoán. Chưa có kết quả Phần 4–5.

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — có kết quả Phần 0–3 | ✓ |
| [`lab1_report.json`](lab1_report.json) — kết quả hiện tại | ✓ |
| `hand_calc.jpg` — ảnh tính tay có các bước | Chưa có |

## 2. Kết quả chính

Dữ liệu sau bỏ trùng: 5.169 tin; tập huấn luyện/xác thực/kiểm tra gồm 3.619/775/775 tin. Tập kiểm tra có 94 tin spam; SUB gồm 20 tin spam và 20 tin ham.

| Phương án | precision | recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 1.000 | 0.840 | 0.913 | 0.919 | 6.282 | 9584.337 | 0.047 | 0.148 | 0.000 |
| 2. BoW + MLP (deep learning) | 1.000 | 0.904 | 0.950 | 0.974 | 11.818 | 23899.847 | 0.412 | 8.791 | 0.000 |
| 3a. Encoder (MiniLM) + kNN | 0.846 | 0.936 | 0.889 | 0.950 | 149.468 | 16.734 | 91.474 | 135.005 | 0.000 |
| 3b. Encoder (MiniLM) + LogReg | 0.945 | 0.915 | 0.930 | 0.974 | 150.896 | 44.828 | 43.999 | 57.615 | 0.000 |
| 3c. Encoder (MiniLM) + MLP head | 0.924 | 0.904 | 0.914 | 0.974 | 150.515 | 46.815 | 28.961 | 44.503 | 0.000 |

Chưa có kết quả LLM. Cột USD/1k tin chỉ tính chi phí API.

## 3. Dự đoán và thực tế

| Thí nghiệm | Dự đoán | Kết quả và giải thích |
| --- | --- | --- |
| `ml_lr` | lr=50 dao động, không hội tụ | Loss dao động mạnh ban đầu rồi giảm về 0.005. Dự đoán chỉ đúng ở giai đoạn đầu; lr lớn vẫn hoạt động được trên dữ liệu này. |
| `dl_curve` | MLP kém LogReg vì dễ overfit | F1 MLP 0.950, cao hơn LogReg (0.913), dù val loss tăng sau chỉ số epoch 6. Dấu hiệu overfit chưa khiến F1 kém LogReg. |
| `dl_relu` | Bỏ ReLU, F1 về gần LogReg | F1 đạt 0.944, gần MLP có ReLU (0.950) hơn LogReg (0.913). Hai lớp Linear không ReLU tương đương một lớp Linear, nhưng cách tối ưu và số bước train khác nhau nên F1 không nhất thiết bằng nhau. |

## 4. Phân tích 50 case khó

Chưa thực hiện.

## 5. Thử thách cá nhân

Đề theo SEED 2018: so sánh MLP có 4 và 512 nơ-ron ẩn về số tham số, F1 test và thời gian train. Chưa thực hiện.

## 6. Phương án cho sản phẩm thật

Tạm chọn BoW + MLP: F1 test 0.950, p95 8.791 ms, thông lượng cả lô khoảng 23.900 tin/giây. Trong phép đo hiện tại, mô hình đạt p95 < 50 ms và vượt mức trung bình khoảng 116 tin/giây của 10 triệu SMS/ngày.

Đánh đổi: p95 cao hơn BoW + LogReg (0.148 ms), nhưng recall tăng từ 0.840 lên 0.904. Chưa có kết quả 50 case khó và thử tải thực tế để chốt phương án; chưa có số liệu chi phí LLM/tháng.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| TODO 2.2 | Có dùng PyTorch không, điền cell thế nào? | Giải thích và cung cấp 5 dòng huấn luyện | Đã dùng đoạn mã trong cell; ô kiểm tra 2.2 đạt. |
| Rà soát | Kiểm tra lỗi trong bài làm | Sửa điều kiện chia cho 0 của recall và dấu `...` làm dự đoán bị từ chối | Đã kiểm tra 5 trường hợp biên của metrics; chưa chạy lại toàn bộ trên Colab. |
| Báo cáo | Đọc hai file bài làm, điền báo cáo ngắn | Tổng hợp số liệu, hỗ trợ diễn đạt và phân tích kết quả | Các giá trị trong bảng khớp với JSON. |

MLP cải thiện recall trên BoW từ 0.840 lên 0.904. Trong thí nghiệm 3.B, MiniLM + LogReg đúng 5/6 tin, BoW + LogReg đúng 4/6; tuy nhiên, các phương án dùng MiniLM đều có độ trễ cao hơn BoW trong lần đo này.
