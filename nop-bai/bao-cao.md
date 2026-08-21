# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Văn Hưng |
| MSSV | 2A202601284 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/NguyenVanHung1707/Track2_Day21_2A202601284_NguyenVanHung |
| Ngày nộp | 21/08/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.10 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.10 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ siêu tham số ở Lần 3 đạt F1-score cao nhất trên lớp dương (0.7149), vượt qua ngưỡng kiểm định chất lượng 0.65. Mặc dù Lần 1 đạt Accuracy cao nhất (0.8780), nhưng F1 của Lần 1 (0.7109) lại thấp hơn Lần 3, chứng minh Accuracy cao dễ gây ngộ nhận về năng lực phân loại thực tế trên tập dữ liệu mất cân bằng. Sự kết hợp giữa `n_estimators=200` và `max_depth=5` giúp mô hình Gradient Boosting khai thác sâu không gian đặc trưng phi tuyến và phát hiện tốt hơn các trường hợp thu nhập cao.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult bị mất cân bằng lớp nghiêm trọng khi lớp thu nhập cao (>50K) chỉ chiếm 24.8%. Một mô hình vô dụng luôn dự đoán toàn bộ là "thu nhập thấp" vẫn đạt Accuracy 75.2% nhưng có F1-score bằng 0 do bỏ sót toàn bộ người có thu nhập cao. F1-score của lớp dương là trung bình điều hòa giữa Precision và Recall, đo lường chính xác khả năng nhận diện đúng các trường hợp thiểu số quan trọng mà không bị lớp đa số làm lu mờ. Ta không dùng average="weighted" hay "macro" vì trọng số lớp đa số sẽ kéo chỉ số lên cao gây ảo tưởng về chất lượng thực sự của mô hình.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lệnh tạo Service Account Key bị chặn | Chính sách bảo mật GCP chặn tạo key JSON thủ công (`disableServiceAccountKeyCreation`) | Cấp IAM `roles/storage.objectAdmin` cho Compute SA và dùng ADC JSON cho GitHub Actions |
| Lỗi unpickle model trên VM | Khác biệt phiên bản scikit-learn giữa môi trường train và môi trường VM | Cài đặt cố định `scikit-learn==1.4.2` đồng nhất theo `requirements.txt` trên VM |
| Lỗi xác thực project khi upload artifact | `storage.Client` trên runner không tự nhận diện `project_id` nếu không truyền tham số | Truyền biến `GOOGLE_CLOUD_PROJECT` và khởi tạo `storage.Client(project=project_id)` |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Khi tăng gấp đôi dữ liệu huấn luyện từ 22.361 lên 44.722 mẫu (cùng phân phối), F1-score tăng nhẹ từ 0.7149 lên 0.7354 (+0.0205) và Accuracy tăng từ 0.8740 lên 0.8820 (+0.0080). Dữ liệu bổ sung giúp thuật toán Gradient Boosting củng cố các biên phân loại và giảm phương sai ước lượng, qua đó nhận diện nhóm thu nhập cao chính xác hơn mà vẫn đảm bảo vượt Quality Gate. Quan trọng nhất, toàn bộ chu trình huấn luyện lại, kiểm tra chất lượng và tái triển khai lên Cloud VM đã diễn ra hoàn toàn tự động chỉ từ một commit cập nhật dữ liệu.

---

## 5. Phần Bonus Đã Thực Hiện

- [x] **Bonus 1 - Tracking MLflow từ xa với DagsHub:** Tích hợp remote server tại `https://dagshub.com/NguyenVanHung1707/Track2_Day21_2A202601284_NguyenVanHung.mlflow`, tự động log tham số và metrics mỗi lần chạy CI/CD.
- [x] **Bonus 2 - Điều chỉnh ngưỡng quyết định:** Quét ngưỡng từ 0.1 đến 0.9, tìm được ngưỡng tối ưu 0.30 giúp F1-score tăng từ 0.7354 lên 0.7537 (+0.0183).
- [x] **Bonus 3 - Báo cáo precision / recall tự động:** Tự động sinh `outputs/detail.txt` chứa Confusion Matrix và chỉ số từng lớp. Với bài toán này, bỏ sót người thu nhập cao (Recall thấp) tốn kém hơn do đánh mất khách hàng tiềm năng giá trị.
- [x] **Bonus 4 - Hoàn trả về phiên bản trước (Rollback Safety):** Tự động sao lưu model hiện tại vào `artifacts/previous/` và Quality Gate sẽ hủy release nếu F1 mới bị suy giảm chất lượng.
- [x] **Bonus 5 - Cảnh báo lệch lạc dữ liệu (Data Drift):** Kiểm tra tỷ lệ nhãn dương trong tập train mới đạt 24.78%, ổn định sát mức tham chiếu 24.80% (độ lệch 0.02% < 5%).
