import os
import numpy as np
import matplotlib.pyplot as plt


def plot_model_comparison():

    # =========================
    # 1. 模型名称
    # =========================
    models = ["KNN", "SVM", "Logistic Regression", "Random Forest"]

    # =========================
    # 2. 指标数据（替换成你自己的结果）
    # =========================
    accuracy = [0.9211, 0.9320, 0.9272, 0.9214]
    precision = [0.87, 0.88, 0.86, 0.87]
    recall = [0.88, 0.89, 0.88, 0.88]
    f1 = [0.86, 0.88, 0.87, 0.87]

    # =========================
    # 3. 位置设置
    # =========================
    x = np.arange(len(models))
    width = 0.2

    # =========================
    # 4. 画图
    # =========================
    plt.figure(figsize=(10, 6))

    plt.bar(x - 1.5 * width, accuracy, width, label="Accuracy")
    plt.bar(x - 0.5 * width, precision, width, label="Precision")
    plt.bar(x + 0.5 * width, recall, width, label="Recall")
    plt.bar(x + 1.5 * width, f1, width, label="F1-score")

    # =========================
    # 5. 美化
    # =========================
    plt.xticks(x, models)
    plt.ylim(0.80, 1.00)
    plt.ylabel("Score")
    plt.title("Performance Comparison of Four Models")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # =========================
    # 6. 自动创建保存目录（关键修复）
    # =========================
    base_dir = os.path.dirname(os.path.dirname(__file__))
    save_dir = os.path.join(base_dir, "results")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "four_models_comparison.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"✔ 图已保存到: {save_path}")


if __name__ == "__main__":
    plot_model_comparison()