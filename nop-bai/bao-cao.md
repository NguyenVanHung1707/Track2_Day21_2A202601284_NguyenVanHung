# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

<!--
HƯỚNG DẪN - đọc rồi XÓA TOÀN BỘ các khối chú thích này sau khi điền xong:

  - Giới hạn: KHÔNG QUÁ 1 TRANG A4, tương đương khoảng 450 - 550 từ nội dung.
  - Chỉ điền vào các chỗ ___ và các ô trong bảng. Không thêm mục mới.
  - Viết bằng câu hoàn chỉnh, không gạch đầu dòng cụt lủn.
  - Kiểm tra độ dài sau khi đã xóa hết chú thích:
        wc -w nop-bai/bao-cao.md
    và xem trước bản in bằng cách mở file trên GitHub rồi Ctrl+P / Cmd+P.
-->

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

**Lý do:** Bộ siêu tham số ở Lần 3 đạt F1-score cao nhất trên lớp dương (0.7149), vượt qua ngưỡng kiểm định 0.65. Mặc dù Lần 1 đạt accuracy cao nhất (0.8780), nhưng F1 của Lần 1 (0.7109) lại thấp hơn Lần 3, chứng minh accuracy cao dễ gây ngộ nhận về năng lực phân loại thực tế. Việc kết hợp `n_estimators=200` và `max_depth=5` giúp mô hình Gradient Boosting học sâu và phát hiện tốt hơn các trường hợp thu nhập cao.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult bị mất cân bằng lớp nghiêm trọng khi lớp thu nhập cao (>50K) chỉ chiếm 24.8%. Một mô hình vô dụng luôn dự đoán "thu nhập thấp" vẫn đạt Accuracy 75.2% nhưng có F1-score bằng 0 do bỏ sót toàn bộ người có thu nhập cao. F1-score của lớp dương là trung bình điều hòa giữa Precision và Recall, đo lường chính xác khả năng nhận diện đúng các trường hợp thiểu số mà không bị lớp đa số làm lu mờ. Ta không dùng average="weighted" hay "macro" vì trọng số lớp đa số sẽ kéo chỉ số lên cao gây ảo tưởng chất lượng mô hình.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lệnh tạo Service Account Key bị chặn | Chính sách bảo mật GCP chặn tạo key JSON thủ công (`disableServiceAccountKeyCreation`) | Cấp IAM `roles/storage.objectAdmin` cho Compute SA và dùng ADC JSON cho GitHub Actions |
| Lỗi unpickle model trên VM | Khác biệt phiên bản scikit-learn giữa môi trường train và môi trường VM | Cài đặt cố định `scikit-learn==1.4.2` đồng nhất theo `requirements.txt` trên VM |
| Lỗi xác thực project khi upload artifact | `storage.Client` trên runner không tự nhận diện `project_id` nếu không truyền tham số | Truyền biến `GOOGLE_CLOUD_PROJECT` và khởi tạo `storage.Client(project=project_id)` |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

<!-- Lấy số liệu từ bảng ở mục 3.6 của tasks/buoc-3.md. -->

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | ___ | ___ |

**Nhận xét:** ___

<!--
Một câu trả lời trung thực kiểu "f1 giảm 0,01 vì dữ liệu mới cùng phân phối, không mang
thêm thông tin mới" được đánh giá cao hơn kết luận sai rằng thêm dữ liệu luôn tốt hơn.
-->

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

<!-- Xóa cả mục 5 nếu không làm bonus. Mỗi bonus tối đa 1 dòng. -->

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: ___
- [ ] Bonus 2 - Điều chỉnh ngưỡng quyết định: ___
- [ ] Bonus 3 - Báo cáo precision / recall tự động: ___
- [ ] Bonus 4 - Hoàn trả về phiên bản trước: ___
- [ ] Bonus 5 - Cảnh báo lệch lạc dữ liệu: ___
