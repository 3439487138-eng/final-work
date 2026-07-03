import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 项目根目录与结果目录
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# 2. 三个模型的准确率
#    这里用你目前已经跑出来的结果
# =========================
def get_model_results():
    data = {
        "Model": ["KNN", "SVM", "Logistic Regression"],
        "Accuracy": [
            0.8772378516624041,   # KNN
            0.8889294848374132,   # SVM
            0.8779685787358421    # Logistic Regression
        ]
    }
    df = pd.DataFrame(data)
    return df


# =========================
# 3. 保存结果表
# =========================
def save_results_table(df):
    save_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"对比表已保存到: {save_path}")


# =========================
# 4. 升级版对比图
# =========================
def plot_model_comparison_advanced(df):
    # 按准确率从高到低排序，图更直观
    df = df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    best_model = df.loc[0, "Model"]
    best_acc = df.loc[0, "Accuracy"]

    plt.figure(figsize=(10, 6))

    # 画柱状图
    bars = plt.bar(df["Model"], df["Accuracy"], edgecolor="black", linewidth=1.2)

    # 自动高亮最佳模型
    for i, bar in enumerate(bars):
        if df.loc[i, "Model"] == best_model:
            bar.set_linewidth(2.0)
            bar.set_hatch("//")

    # 标题与坐标轴
    plt.title("Accuracy Comparison of Three Models", fontsize=16, fontweight="bold")
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)

    # 为了让差异更明显，设置y轴范围
    y_min = min(df["Accuracy"]) - 0.01
    y_max = max(df["Accuracy"]) + 0.01
    plt.ylim(y_min, y_max)

    # 网格线
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # 在柱子上方标注准确率
    for bar, acc in zip(bars, df["Accuracy"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    # 在图中添加最佳模型说明
    plt.text(
        0.02, 0.95,
        f"Best Model: {best_model}\nAccuracy: {best_acc:.4f}",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "model_accuracy_comparison_advanced.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"升级版对比图已保存到: {save_path}")


# =========================
# 5. 可选：画折线对比图
#    这个图适合补充展示
# =========================
def plot_model_line_chart(df):
    df = df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(9, 5))
    plt.plot(df["Model"], df["Accuracy"], marker="o", linewidth=2)

    plt.title("Model Accuracy Trend", fontsize=15, fontweight="bold")
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(min(df["Accuracy"]) - 0.01, max(df["Accuracy"]) + 0.01)
    plt.grid(True, linestyle="--", alpha=0.6)

    for x, y in zip(df["Model"], df["Accuracy"]):
        plt.text(x, y + 0.0005, f"{y:.4f}", ha="center", fontsize=10)

    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "model_accuracy_line_chart.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"折线对比图已保存到: {save_path}")


# =========================
# 6. 主函数
# =========================
def main():
    df = get_model_results()
    print(df)

    save_results_table(df)
    plot_model_comparison_advanced(df)
    plot_model_line_chart(df)


if __name__ == "__main__":
    main()