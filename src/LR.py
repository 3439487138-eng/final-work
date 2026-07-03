import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import seaborn as sns


# =========================
# 1. 数据读取
# =========================
def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data/processed")

    train = pd.read_csv(os.path.join(data_dir, "train_processed.csv"))
    val = pd.read_csv(os.path.join(data_dir, "val_processed.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test_processed.csv"))

    X_train = train.drop(columns=["Class"])
    y_train = train["Class"]

    X_val = val.drop(columns=["Class"])
    y_val = val["Class"]

    X_test = test.drop(columns=["Class"])
    y_test = test["Class"]

    return X_train, y_train, X_val, y_val, X_test, y_test


# =========================
# 2. 训练 LR（已修复multi_class问题）
# =========================
def train_lr():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # ✔ 新版 sklearn 不需要 multi_class
    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    )

    model.fit(X_train, y_train)

    # 验证集
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    print("\n===== Logistic Regression =====")
    print("Validation Accuracy:", val_acc)

    # 测试集
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print("Test Accuracy:", test_acc)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_test_pred))

    return y_test, y_test_pred, test_acc


# =========================
# 3. 混淆矩阵可视化
# =========================
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Oranges", fmt="d")

    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    base_dir = os.path.dirname(os.path.dirname(__file__))
    save_path = os.path.join(base_dir, "results", "lr_confusion_matrix.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# =========================
# 4. 主函数
# =========================
def main():
    y_test, y_pred, acc = train_lr()
    plot_confusion_matrix(y_test, y_pred)


if __name__ == "__main__":
    main()