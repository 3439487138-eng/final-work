import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# 1. 解决路径问题（核心修复）
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "data/processed/train_processed.csv")
TEST_PATH  = os.path.join(BASE_DIR, "data/processed/test_processed.csv")


# =========================
# 2. 读取数据
# =========================
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    X_train = train.drop(columns=["Class"])
    y_train = train["Class"]

    X_test = test.drop(columns=["Class"])
    y_test = test["Class"]

    return X_train, y_train, X_test, y_test


# =========================
# 3. 保存结果
# =========================
def save_results(acc, report, cm):
    save_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(save_dir, exist_ok=True)

    # 保存文本报告
    with open(os.path.join(save_dir, "knn_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write(report)

    # 保存混淆矩阵
    pd.DataFrame(cm).to_csv(
        os.path.join(save_dir, "knn_confusion_matrix.csv"),
        index=False
    )


# =========================
# 4. 可视化混淆矩阵
# =========================
def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("KNN Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    save_path = os.path.join(BASE_DIR, "results", "knn_confusion_matrix.png")
    plt.savefig(save_path)
    plt.show()


# =========================
# 5. 训练KNN
# =========================
def run_knn():
    X_train, y_train, X_test, y_test = load_data()

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("========== KNN RESULT ==========")
    print("Accuracy:", acc)
    print(report)

    # 保存 + 可视化
    save_results(acc, report, cm)
    plot_confusion_matrix(cm)


# =========================
# 6. 主入口
# =========================
if __name__ == "__main__":
    run_knn()