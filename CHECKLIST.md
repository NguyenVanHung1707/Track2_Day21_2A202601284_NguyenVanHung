# CHECKLIST TOÀN BỘ CÔNG VIỆC CẦN THỰC HIỆN
## Lab MLOps: Từ Thực Nghiệm Cục Bộ Đến Triển Khai Liên Tục (Day 21 - VinAI / VinUni)

> **Mục tiêu**: Xây dựng hoàn chỉnh hệ thống MLOps gồm: Quản lý thí nghiệm (MLflow), Quản lý dữ liệu (DVC + Cloud Storage), CI/CD Pipeline (GitHub Actions), Phục vụ mô hình (FastAPI trên Cloud VM) và Tự động huấn luyện liên tục (Continuous Training).

---

## 📋 MỤC LỤC
1. [Chuẩn Bị Môi Trường & Khởi Tạo Dự Án](#1-chuẩn-bị-môi-trường--khởi-tạo-dự-án)
2. [Bước 1: Thực Nghiệm Cục Bộ & MLflow Tracking](#2-bước-1-thực-nghiệm-cục-bộ--mlflow-tracking)
3. [Bước 2: Pipeline CI/CD Tự Động & Triển Khai Cloud](#3-bước-2-pipeline-cicd-tự-động--triển-khai-cloud)
4. [Bước 3: Huấn Luyện Liên Tục Khi Có Dữ Liệu Mới](#4-bước-3-huấn-luyện-liên-tục-khi-có-dữ-liệu-mới)
5. [Thách Thức Nâng Cao - Bonus (Tùy chọn - Tối đa 20đ)](#5-thách-thức-nâng-cao---bonus-tùy-chọn---tối-đa-20đ)
6. [Thu Thập Bằng Chứng & Ảnh Chụp Màn Hình](#6-thu-thập-bằng-chứng--ảnh-chụp-màn-hình)
7. [Hoàn Thiện Báo Cáo (`nop-bai/bao-cao.md`)](#7-hoàn-thiện-báo-cáo-nop-baibao-caomd)
8. [Kiểm Tra Cuối Cùng & Nộp Bài](#8-kiểm-tra-cuối-cùng--nộp-bài)

---

## 1. Chuẩn Bị Môi Trường & Khởi Tạo Dự Án

- [x] **1.1. Kiểm tra phần mềm cài đặt trên máy cá nhân**:
  - [x] Python `>= 3.10` (`python --version` -> Python 3.11.2)
  - [x] Git (`git --version` -> Git 2.33.0)
  - [x] Cloud CLI (`gcloud --version` -> Google Cloud SDK 580.0.0, gsutil 5.37)
- [x] **1.2. Khởi tạo môi trường ảo Python & cài đặt dependencies**:
  - [x] Tạo venv: `python -m venv .venv`
  - [x] Kích hoạt venv & cài thư viện: `pip install -r requirements.txt` (MLflow 2.13.0, DVC[gs] 3.50.1, google-cloud-storage 2.16.0, scikit-learn 1.4.2, FastAPI 0.111.0, pytest 8.2.0...)
- [x] **1.3. Chuẩn bị tập dữ liệu Adult Income**:
  - [x] Chạy lệnh: `python prepare_data.py`
  - [x] Kiểm tra các file được tạo trong `data/`:
    - [x] `data/train_batch1.csv` (22.361 mẫu)
    - [x] `data/holdout.csv` (500 mẫu)
    - [x] `data/train_batch2.csv` (22.361 mẫu)
- [x] **1.4. Kiểm tra cấu hình `.gitignore`**:
  - [x] Đảm bảo các file dữ liệu thô (`data/*.csv`), database MLflow (`mlflow.db`), artifacts, models, venv, và key (`sa-key.json`) không bị commit vào git.

---

## 2. Bước 1: Thực Nghiệm Cục Bộ & MLflow Tracking

- [x] **2.1. Cấu hình biến môi trường MLflow**:
  - [x] Đặt `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`
  - [x] Đặt `MLFLOW_ARTIFACT_ROOT=./mlartifacts`
- [x] **2.2. Khởi tạo file cấu hình siêu tham số `params.yaml`**:
  - [x] Điền các siêu tham số cho `GradientBoostingClassifier` (`n_estimators`, `learning_rate`, `max_depth`).
- [x] **2.3. Lập trình mã nguồn huấn luyện `src/train.py`**:
  - [x] `TODO 1.6.1` / `TODO 1`: Đọc dữ liệu từ `data_path` và `eval_path` bằng pandas.
  - [x] `TODO 1.6.2` / `TODO 2`: Tách đặc trưng ($X$) và nhãn ($y$) (loại bỏ cột `target`).
  - [x] `TODO 1.6.3`: Mở MLflow run (`with mlflow.start_run():`).
  - [x] `TODO 1.6.4` / `TODO 3`: Ghi nhận siêu tham số (`mlflow.log_params(params)`).
  - [x] `TODO 1.6.5` / `TODO 4`: Khởi tạo & fit `GradientBoostingClassifier(**params, random_state=42)`.
  - [x] `TODO 1.6.6` / `TODO 5`: Dự đoán trên holdout và tính `f1_score(y_eval, preds)` (cho lớp dương target=1, không dùng weighted/macro) và `accuracy_score(y_eval, preds)`.
  - [x] `TODO 1.6.7` / `TODO 6`: Ghi nhận metrics lên MLflow (`f1_score`, `accuracy`).
  - [x] `TODO 1.6.8`: Log mô hình vào MLflow artifact (`mlflow.sklearn.log_model(model, "model")`).
  - [x] `TODO 1.6.9` / `TODO 7`: In kết quả ra màn hình (`F1` và `Accuracy`).
  - [x] `TODO 1.6.10` / `TODO 8`: Lưu kết quả đánh giá vào `outputs/report.json` dạng `{"f1_score": f1, "accuracy": acc}`.
  - [x] `TODO 1.6.11` / `TODO 9`: Lưu mô hình huấn luyện vào `models/model.joblib`.
  - [x] `TODO 1.6.12` / `TODO 10`: Trả về giá trị `f1` từ hàm `train()`.
- [x] **2.4. Thực hiện ít nhất 3 lần chạy thí nghiệm với các bộ siêu tham số khác nhau**:
  - [x] Lần 1 (`n_estimators: 100`, `learning_rate: 0.1`, `max_depth: 3`): `F1: 0.7109 | Accuracy: 0.8780`
  - [x] Lần 2 (`n_estimators: 50`, `learning_rate: 0.05`, `max_depth: 2`): `F1: 0.6051 | Accuracy: 0.8460`
  - [x] Lần 3 (`n_estimators: 200`, `learning_rate: 0.1`, `max_depth: 5`): `F1: 0.7149 | Accuracy: 0.8740`
  - [x] Lần 4 (`n_estimators: 100`, `learning_rate: 0.15`, `max_depth: 4`): `F1: 0.7064 | Accuracy: 0.8720`
- [x] **2.5. Phân tích kết quả trên MLflow UI & Chọn Best Model**:
  - [x] So sánh các runs theo `f1_score` và `accuracy`.
  - [x] Chọn bộ tham số tốt nhất Lần 3 (`n_estimators: 200`, `learning_rate: 0.1`, `max_depth: 5`) đạt `f1_score = 0.7149 >= 0.65` và lưu cố định vào `params.yaml`.
  - [x] Đã ghi số liệu & phân tích vào [Mục 1 & 2 của báo cáo](nop-bai/bao-cao.md).
  - [x] Mở MLflow UI (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) chụp ảnh `01-mlflow-ui.png` lưu vào `nop-bai/anh-chup-man-hinh/`.

---

## 3. Bước 2: Pipeline CI/CD Tự Động & Triển Khai Cloud

- [x] **3.1. Thiết lập Cloud Infrastructure & Cloud Storage**:
  - [x] Tạo Bucket trên Cloud Storage (`gs://income-lab-2a202601284`).
  - [x] Tạo Service Account / IAM Credentials với quyền `roles/storage.objectAdmin` trên bucket.
  - [x] Xuất file credential (ví dụ `sa-key.json` / ADC cho GCP) và lưu bảo mật (không commit vào Git).
- [x] **3.2. Cấu hình DVC Remote & Đẩy Dữ Liệu**:
  - [x] Khởi tạo DVC: `dvc init`
  - [x] Thêm cloud remote: `dvc remote add -d gcs-remote gs://income-lab-2a202601284/dvc`
  - [x] Cấu hình credential: GCP ADC / Service Account
  - [x] Đưa các file dataset vào DVC tracking:
    - [x] `dvc add data/train_batch1.csv`
    - [x] `dvc add data/holdout.csv`
    - [x] `dvc add data/train_batch2.csv`
  - [x] Commit file con trỏ DVC (`.dvc`) và `.dvc/config` vào Git.
  - [x] Đẩy dữ liệu lên Cloud Storage: `dvc push` (xác nhận folder `dvc/` đã xuất hiện trên cloud).
- [x] **3.3. Thiết lập Cloud VM (GCE / EC2 / Azure VM)**:
  - [x] Tạo máy ảo Ubuntu 22.04 LTS (`income-api`, IP `34.60.69.202`).
  - [x] Cấu hình Firewall mở cổng `8080` (TCP).
  - [x] Lấy địa chỉ Public IP của VM (`SERVER_HOST = 34.60.69.202`).
  - [x] SSH vào VM, cài đặt môi trường: `fastapi`, `uvicorn`, `scikit-learn==1.4.2`, `joblib`, `google-cloud-storage`.
  - [x] Tạo các thư mục `~/models`, `~/src` trên VM.
  - [x] Cấu hình quyền Cloud Storage cho VM Service Account.
- [x] **3.4. Lập trình REST API `src/serve.py`**:
  - [x] `TODO 2.6.1` -> `2.6.5`: Viết hàm `download_model()` tải `model.joblib` từ Cloud Storage (`artifacts/current/model.joblib`) về `~/models/model.joblib`.
  - [x] `TODO 2.6.6`: Viết endpoint `GET /healthz` trả về `{"status": "ok"}`.
  - [x] `TODO 2.6.7`: Kiểm tra đầu vào `len(req.features) == 10` (nếu sai trả lỗi 400).
  - [x] `TODO 2.6.8`: Gọi `model.predict([req.features])`.
  - [x] `TODO 2.6.9`: Trả về kết quả JSON `{"prediction": <0|1>, "label": <"thu_nhap_thap"|"thu_nhap_cao">}`.
  - [x] Copy `src/serve.py` lên VM (`~/src/serve.py`).
- [x] **3.5. Cấu hình Systemd Service trên VM**:
  - [x] Tạo file `/etc/systemd/system/income-api.service` với biến môi trường `ARTIFACT_BUCKET` và credentials.
  - [x] Reload daemon và enable service: `sudo systemctl daemon-reload && sudo systemctl enable income-api`.
- [x] **3.6. Cấu hình SSH Deploy & GitHub Secrets**:
  - [x] Tạo cặp SSH key chuyên dụng: `ssh-keygen -t ed25519 -f ~/.ssh/income_deploy -N "" -C "github-actions-deploy"`.
  - [x] Thêm public key (`income_deploy.pub`) vào `~/.ssh/authorized_keys` trên VM.
  - [x] Thêm đủ 5 Secrets trên GitHub Repo (Settings > Secrets and variables > Actions):
    - [x] `STORAGE_CREDENTIALS` (Nội dung JSON service account / ADC)
    - [x] `ARTIFACT_BUCKET` (`income-lab-2a202601284`)
    - [x] `SERVER_HOST` (`34.60.69.202`)
    - [x] `SERVER_USER` (`HUNG`)
    - [x] `SERVER_SSH_KEY` (Nội dung private key `income_deploy`)
- [x] **3.7. Viết Unit Tests `tests/test_train.py`**:
  - [x] `TODO 2.10.1` -> `2.10.5`: Viết hàm `_make_temp_data(tmp_path)` sinh dữ liệu giả lập 10 features, 2 classes (160 train, 40 holdout).
  - [x] `TODO 2.10.6` -> `2.10.7`: Viết `test_train_returns_float` kiểm tra `train()` trả về float trong $[0.0, 1.0]$.
  - [x] `TODO 2.10.8` -> `2.10.9`: Viết `test_report_file_created` kiểm tra file `outputs/report.json` chứa `f1_score` và `accuracy`.
  - [x] `TODO 2.10.10`: Viết `test_model_file_created` kiểm tra file `models/model.joblib` được tạo.
  - [x] Chạy kiểm thử cục bộ: `pytest tests/ -v` (3/3 test pass).
- [x] **3.8. Hoàn thiện Pipeline CI/CD `.github/workflows/cicd.yml`**:
  - [x] `TODO 2.11.1`: Job `unit-test` chạy `pytest tests/ -v`.
  - [x] `TODO 2.11.2`: Job `train` cấu hình xác thực Cloud Storage từ secret.
  - [x] `TODO 2.11.3`: Job `train` kéo dữ liệu bằng `dvc pull`.
  - [x] `TODO 2.11.4`: Job `train` đọc `f1_score` từ `outputs/report.json` và xuất ra `$GITHUB_OUTPUT`.
  - [x] `TODO 2.11.5`: Job `train` upload `models/model.joblib` lên Cloud Storage (`artifacts/current/model.joblib`).
  - [x] `TODO 2.11.6`: Job `quality-gate` kiểm tra `float(f1) >= 0.65` (nếu không đạt thì exit 1 chặn release).
  - [x] `TODO 2.11.7` -> `2.11.8`: Job `release` dùng `appleboy/ssh-action` restart service trên VM và kiểm tra `curl http://localhost:8080/healthz`.
- [x] **3.9. Kích hoạt & Kiểm tra Pipeline Bước 2**:
  - [x] Commit và push code lên GitHub `main`.
  - [x] Theo dõi Actions tab: Đảm bảo cả 4 jobs (`Unit Test`, `Train`, `Quality Gate`, `Release`) đều XANH.
  - [x] Kiểm tra thủ công API qua terminal:
    - [x] `curl http://<VM_IP>:8080/healthz`
    - [x] `curl -X POST http://<VM_IP>:8080/score -H "Content-Type: application/json" -d '{"features": [60, 2, 5, 2, 4, 0, 1, 0, 0, 45]}'`
    - [x] `curl -X POST http://<VM_IP>:8080/score -H "Content-Type: application/json" -d '{"features": [28, 2, 14, 2, 11, 0, 1, 0, 0, 45]}'`
  - [x] Chụp 3 ảnh bằng chứng Bước 2:
    - [x] `02-actions-buoc-2.png`
    - [x] `04-curl-api.png`
    - [x] `05-cloud-storage.png`

---

## 4. Bước 3: Huấn Luyện Liên Tục Khi Có Dữ Liệu Mới

- [x] **4.1. Bổ sung dữ liệu mới**:
  - [x] Chạy script: `python append_batch.py` (ghép `train_batch2.csv` vào `train_batch1.csv`).
  - [x] Kiểm tra số dòng: `wc -l data/train_batch1.csv` (kết quả 44.723 dòng gồm header).
- [x] **4.2. Cập nhật DVC & Kích hoạt CI/CD**:
  - [x] Thông báo cho DVC: `dvc add data/train_batch1.csv`
  - [x] Commit file `.dvc` vào git: `git add data/train_batch1.csv.dvc && git commit -m "data: bổ sung 22361 mẫu dữ liệu mới (train_batch2)"`
  - [x] Đẩy dữ liệu mới lên Cloud Storage trước: `dvc push`
  - [x] Đẩy git commit lên GitHub: `git push origin main`
- [x] **4.3. Giám sát Pipeline Tự Động Phản Ứng**:
  - [x] Kiểm tra trên tab Actions: Pipeline tự kích hoạt với đúng commit message dữ liệu.
  - [x] Đảm bảo 4 jobs hoàn thành thành công và mô hình mới được tự động cập nhật lên VM.
- [x] **4.4. Đánh giá & So sánh kết quả**:
  - [x] Tải `outputs/report.json` từ artifact của Bước 2 và Bước 3.
  - [x] Ghi nhận số liệu `f1_score` và `accuracy` giữa 2 bước vào bảng so sánh.
  - [x] Gọi thử lại API trên VM để xác nhận mô hình mới đang phục vụ.
  - [x] Chụp ảnh `03-actions-buoc-3.png`.

---

## 5. Thách Thức Nâng Cao - Bonus (Tùy chọn - Tối đa 20đ)

- [x] **Bonus 1: Tracking MLflow Từ Xa Với DagsHub (+4đ)**: Tích hợp remote tracking server tại DagsHub, tự động log metrics & model qua GitHub Actions.
- [x] **Bonus 2: Điều Chỉnh Ngưỡng Quyết Định Tối Ưu (+4đ)**: Quét threshold 0.1-0.9, tìm ngưỡng tối ưu 0.30 nâng F1 lên 0.7537.
- [x] **Bonus 3: Báo Cáo Precision / Recall Tự Động (+4đ)**: Xuất `outputs/detail.txt` chứa Confusion Matrix và Classification Report.
- [x] **Bonus 4: Cơ Chế Rollback / An Toàn Trước Khi Release (+4đ)**: Tự động sao lưu model hiện tại vào `artifacts/previous/` và kiểm tra chống suy giảm chất lượng trước khi release.
- [x] **Bonus 5: Cảnh Báo Lệch Lạc Phân Phối Dữ Liệu (Data Drift) (+4đ)**: Kiểm tra tỷ lệ nhãn dương trong tập train mới so với baseline 24.8%.

---

## 6. Thu Thập Bằng Chứng & Ảnh Chụp Màn Hình

Đảm bảo đủ **5 ảnh bắt buộc** đặt tại `nop-bai/anh-chup-man-hinh/` (đúng tên file, dung lượng mỗi ảnh $< 1\text{MB}$, hiển thị URL trình duyệt):

- [x] `01-mlflow-ui.png`: MLflow UI hiển thị $\ge 3$ runs, thấy rõ các cột `f1_score`, `accuracy`, `n_estimators`, `learning_rate`, `max_depth` và thanh URL `localhost:5000`.
- [x] `02-actions-buoc-2.png`: GitHub Actions Bước 2 với 4 jobs xanh (`Unit Test`, `Train`, `Quality Gate`, `Release`) và commit message code.
- [x] `03-actions-buoc-3.png`: GitHub Actions Bước 3 được kích hoạt bởi commit dữ liệu mới, 4 jobs xanh.
- [x] `04-curl-api.png`: Terminal hiển thị cả 2 lệnh `curl http://<VM_IP>:8080/healthz` và `curl -X POST http://<VM_IP>:8080/score`, thấy rõ IP VM và kết quả JSON.
- [x] `05-cloud-storage.png`: Giao diện Cloud Console hiển thị thư mục `dvc/` và file `artifacts/current/model.joblib`, thấy rõ tên bucket.
- [x] `06-dagshub-mlflow.png`: Giao diện DagsHub Experiments hiển thị remote MLflow tracking cho repo GitHub.
- [x] `07-bonus-mlflow.png`: MLflow UI tracking các độ đo Bonus (`best_threshold: 0.30`, `best_f1_tuned: 0.7537`, `positive_ratio: 0.2478`).
- [x] `08-actions-bonus.png`: GitHub Actions Bonus run với 4 jobs xanh và artifact `model-reports`.

---

## 7. Hoàn Thiện Báo Cáo (`nop-bai/bao-cao.md`)

- [x] **7.1. Điền thông tin cá nhân**: Họ và tên (Nguyễn Văn Hưng), MSSV (2A202601284), Khóa K4, URL Repo GitHub, Ngày nộp.
- [x] **7.2. Mục 1 - Siêu tham số & Lý do**:
  - [x] Bảng kết quả 3 lần chạy MLflow.
  - [x] Bộ tham số tốt nhất đã chọn (`n_estimators: 200`, `learning_rate: 0.1`, `max_depth: 5`).
  - [x] Giải thích lý do chọn dựa trên F1-score và phân tích mối quan hệ đánh đổi giữa `n_estimators` và `learning_rate`.
- [x] **7.3. Mục 2 - Lý do chọn F1 thay vì Accuracy**:
  - [x] Phân tích độ mất cân bằng lớp (24.8% lớp >50K).
  - [x] Giải thích vì sao mô hình đoán bừa đạt Accuracy 75.2% nhưng vô dụng.
  - [x] Giải thích vì sao tính F1 cho lớp dương mà không dùng `weighted` hay `macro`.
- [x] **7.4. Mục 3 - Khó khăn & Cách giải quyết**:
  - [x] Điền 3 khó khăn thực tế gặp phải và giải pháp xử lý.
- [x] **7.5. Mục 4 - So sánh Bước 2 và Bước 3**:
  - [x] Bảng số liệu F1 và Accuracy của 2 bước.
  - [x] Nhận xét chi tiết về sự thay đổi khi tăng gấp đôi dữ liệu.
- [x] **7.6. Dọn dẹp & Định dạng**:
  - [x] **Xóa toàn bộ các khối chú thích hướng dẫn `<!-- ... -->`**.
  - [x] Kiểm tra độ dài báo cáo **không vượt quá 1 trang A4** (~640 từ).

---

## 8. Kiểm Tra Cuối Cùng & Nộp Bài

- [x] **8.1. Kiểm tra mã nguồn & file cấu hình**:
  - [x] Không có file nhạy cảm (`sa-key.json`, `.env`, tokens) bị commit.
  - [x] `requirements.txt` và `params.yaml` đầy đủ.
- [x] **8.2. Push toàn bộ thay đổi lên GitHub**:
  - [x] `git status` sạch sẽ.
  - [x] `git push origin main` hoàn tất.
- [x] **8.3. Kiểm tra tính khả dụng công khai**:
  - [x] GitHub Repository đã được chuyển sang chế độ **Public** (`https://github.com/NguyenVanHung1707/Track2_Day21_2A202601284_NguyenVanHung`).
- [ ] **8.4. Nộp link bài tập**:
  - [ ] Copy URL GitHub Repo.
  - [ ] Dán vào trang nộp bài tại **https://codelabs.vlearn.dev**.

