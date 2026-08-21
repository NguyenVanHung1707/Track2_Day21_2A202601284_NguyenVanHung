import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65

# Cau hinh tracking URI mac dinh neu chua co trong bien moi truong
if not os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # -------------------------------------------------------------------------
    # BONUS 5: Canh bao lech lac phan phoi du lieu (Data Drift Detection)
    # -------------------------------------------------------------------------
    pos_ratio = float((y_train == 1).mean())
    baseline_ratio = 0.248  # Ty le lop duong tham chieu trong Adult dataset (24.8%)
    drift_diff = abs(pos_ratio - baseline_ratio)
    if drift_diff > 0.05:
        print(
            f"[WARNING - DATA DRIFT] Ty le lop duong trong tap train la {pos_ratio:.4f} "
            f"(lech {drift_diff:.4f} so voi chuan {baseline_ratio:.4f}, vuot nguong 0.05)!"
        )
    else:
        print(f"[DATA DRIFT CHECK PASSED] Ty le lop duong: {pos_ratio:.4f} (on dinh quanh {baseline_ratio:.4f})")

    with mlflow.start_run():
        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("eval_path", eval_path)

        # TODO 4: Khoi tao va huan luyen GradientBoostingClassifier
        # random_state=42 de dam bao tinh tai tao
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap holdout va tinh chi so (nguong mac dinh 0.5)
        preds = model.predict(X_eval)
        f1 = float(f1_score(y_eval, preds))
        acc = float(accuracy_score(y_eval, preds))

        # ---------------------------------------------------------------------
        # BONUS 2: Dieu chinh nguong quyet dinh toi uu (Threshold Tuning)
        # ---------------------------------------------------------------------
        probs = model.predict_proba(X_eval)[:, 1]
        threshold_candidates = np.arange(0.1, 0.95, 0.05)
        best_thresh = 0.5
        best_f1_score = f1

        for thresh in threshold_candidates:
            thresh_preds = (probs >= thresh).astype(int)
            thresh_f1 = float(f1_score(y_eval, thresh_preds, zero_division=0))
            if thresh_f1 > best_f1_score:
                best_f1_score = thresh_f1
                best_thresh = float(thresh)

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("positive_ratio", pos_ratio)
        mlflow.log_metric("best_threshold", best_thresh)
        mlflow.log_metric("best_f1_tuned", best_f1_score)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print(f"[BONUS 2 - THRESHOLD TUNING] Best Threshold: {best_thresh:.2f} -> Best F1: {best_f1_score:.4f}")

        # ---------------------------------------------------------------------
        # BONUS 3: Bao cao Precision / Recall & Confusion Matrix (outputs/detail.txt)
        # ---------------------------------------------------------------------
        cm = confusion_matrix(y_eval, preds)
        prec_0 = float(precision_score(y_eval, preds, pos_label=0, zero_division=0))
        rec_0 = float(recall_score(y_eval, preds, pos_label=0, zero_division=0))
        prec_1 = float(precision_score(y_eval, preds, pos_label=1, zero_division=0))
        rec_1 = float(recall_score(y_eval, preds, pos_label=1, zero_division=0))

        detail_text = (
            "===========================================================\n"
            "            CHI TIET DANH GIA MO HINH INCOME\n"
            "===========================================================\n\n"
            f"1. Cac Chi So Chinh:\n"
            f"   - F1-score (Lop duong >50K)   : {f1:.4f}\n"
            f"   - Accuracy (Do chinh xac tong): {acc:.4f}\n"
            f"   - Ty le lop duong tap Train   : {pos_ratio:.4f}\n"
            f"   - Nguong toi uu (Threshold)   : {best_thresh:.2f} (F1 toi uu: {best_f1_score:.4f})\n\n"
            f"2. Chi Tiet Theo Tung Lop:\n"
            f"   - Lop 0 (Thu nhap <=50K)     : Precision = {prec_0:.4f}, Recall = {rec_0:.4f}\n"
            f"   - Lop 1 (Thu nhap >50K)      : Precision = {prec_1:.4f}, Recall = {rec_1:.4f}\n\n"
            f"3. Ma Tran Nhap Nhang (Confusion Matrix):\n"
            f"                  Du doan 0 (<=50K)   Du doan 1 (>50K)\n"
            f"   Thuc te 0:     {cm[0][0]:<19} {cm[0][1]}\n"
            f"   Thuc te 1:     {cm[1][0]:<19} {cm[1][1]}\n\n"
            f"4. Bang Tong Hop Phan Loai (Classification Report):\n"
            f"{classification_report(y_eval, preds, target_names=['<=50K', '>50K'])}\n"
        )

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write(detail_text)

        # TODO 8: Luu metrics ra file outputs/report.json
        report_data = {
            "f1_score": f1,
            "accuracy": acc,
            "positive_ratio": pos_ratio,
            "best_threshold": best_thresh,
            "best_f1_tuned": best_f1_score,
            "precision_class_1": prec_1,
            "recall_class_1": rec_1,
        }
        with open("outputs/report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        # TODO 9: Luu mo hinh ra file models/model.joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # TODO 10: Tra ve f1
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
