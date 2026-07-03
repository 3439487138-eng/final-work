import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# 1. 路径设置
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "data/processed/train_processed.csv")
TEST_PATH = os.path.join(BASE_DIR, "data/processed/test_processed.csv")

RESULT_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULT_DIR, "figures")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# =========================
# 2. 数据读取
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
# 3. 随机森林模型
# =========================
def train_rf():
    X_train, y_train, X_test, y_test = load_data()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\n========== RANDOM FOREST RESULT ==========")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred, zero_division=0))

    # 保存accuracy
    with open(os.path.join(RESULT_DIR, "rf_accuracy.txt"), "w") as f:
        f.write(str(acc))

    return y_test, y_pred, model, acc


# =========================
# 4. 混淆矩阵
# =========================
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    save_path = os.path.join(FIG_DIR, "rf_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("RF混淆矩阵已保存:", save_path)


# =========================
# 5. 特征重要性（论文加分点🔥）
# =========================
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(df["Feature"][:10], df["Importance"][:10])
    plt.xticks(rotation=45)

    plt.title("Random Forest Feature Importance (Top 10)")
    plt.ylabel("Importance")

    save_path = os.path.join(FIG_DIR, "rf_feature_importance.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("RF特征重要性图已保存:", save_path)


# =========================
# 6. Accuracy柱状图
# =========================
def plot_accuracy(acc):
    plt.figure(figsize=(5, 4))

    plt.bar(["Random Forest"], [acc], color="green", edgecolor="black")
    plt.ylim(0.8, 1.0)

    plt.title("Random Forest Accuracy")
    plt.ylabel("Accuracy")

    plt.text(0, acc + 0.002, f"{acc:.4f}", ha="center")

    save_path = os.path.join(FIG_DIR, "rf_accuracy.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("RF accuracy图已保存:", save_path)


# =========================
# 7. 主函数
# =========================
def main():
    y_test, y_pred, model, acc = train_rf()

    plot_confusion_matrix(y_test, y_pred)
    plot_accuracy(acc)

    X_train, _, _, _ = load_data()
    plot_feature_importance(model, X_train.columns)


if __name__ == "__main__":
    main()