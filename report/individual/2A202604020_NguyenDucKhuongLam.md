# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **SEED:** 2018.
- **Tiến độ:** đã có kết quả Phần 0–3; JSON ghi nhận 8/8 kiểm tra đạt và 6/10 dự đoán. Chưa có kết quả Phần 4–5.

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

Chưa có số liệu LLM. USD/1k tin chỉ tính chi phí API.

## 3. Dự đoán và thực tế

| Thí nghiệm | Dự đoán | Thực tế và giải thích |
| --- | --- | --- |
| `ml_lr` | lr=50 dao động, không hội tụ | Hàm mất mát dao động mạnh ban đầu rồi giảm về 0.005, thay vì tiếp tục dao động như dự đoán. |
| `dl_curve` | MLP kém LogReg vì dễ quá khớp | F1 MLP đạt 0.950, cao hơn LogReg (0.913), dù mất mát xác thực tăng sau epoch thứ 7 (chỉ số 6). Dấu hiệu quá khớp chưa khiến F1 kém LogReg. |
| `dl_relu` | Bỏ ReLU, F1 về gần LogReg | F1 đạt 0.944, gần MLP (0.950) hơn LogReg (0.913). Tương đương về dạng biểu diễn nhưng khác cách tối ưu và số bước huấn luyện. |

## 4. Phân tích 50 case khó

Chưa thực hiện.

## 5. Thử thách cá nhân

SEED 2018 ứng với đề #3: so sánh MLP có 4 và 512 nơ-ron ẩn về số tham số, F1 test và thời gian huấn luyện. Chưa thực hiện.

## 6. Phương án cho sản phẩm thật

Tạm chọn BoW + MLP: F1 test 0.950, p95 8.791 ms, thông lượng cả lô khoảng 23.900 tin/giây. Trong thử nghiệm, mô hình đạt ngưỡng p95 < 50 ms và vượt mức trung bình khoảng 116 tin/giây.

Đánh đổi: p95 cao hơn BoW + LogReg (0.148 ms), nhưng recall tốt hơn (0.904 so với 0.840). Cần bổ sung thử nghiệm 50 case khó và kiểm tra tải thực tế; chưa có số liệu chi phí LLM/tháng.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Chuẩn bị | Cách làm trên Colab, file cần nộp, mẫu báo cáo | Giải thích quy trình theo tài liệu repo | Tên repo theo yêu cầu LMS. |
| TODO 2.2 | Có dùng PyTorch không, điền cell thế nào? | Giải thích và cung cấp 5 dòng huấn luyện | Notebook có đoạn mã tương ứng và kết quả kiểm tra 2.2 đạt. |
| Báo cáo | Đọc hai file bài làm, điền báo cáo ngắn | Tổng hợp số liệu và đánh dấu phần thiếu | Đối chiếu JSON với output notebook; phát hiện ô `ml_f1` bị từ chối ghi. |

MLP cải thiện recall trên BoW dù có dấu hiệu quá khớp. Embedding giúp nhận diện một số tin né từ khóa, nhưng chưa vượt BoW + MLP về F1 test và độ trễ trong lần chạy này.
