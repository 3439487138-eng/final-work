import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 输入你的模型结果
# =========================
results = {
    "KNN": {
        "train_acc": 0.935,
        "test_acc": 0.921
    },
    "SVM": {
        "train_acc": 0.948,
        "test_acc": 0.932
    },
    "LR": {
        "train_acc": 0.930,
        "test_acc": 0.927
    },
    "RF": {
        "train_acc": 0.960,
        "test_acc": 0.921
    }
}

# =========================
# 2. 生成表格数据
# =========================
rows = []

for model, scores in results.items():
    train_acc = scores["train_acc"]
    test_acc = scores["test_acc"]
    gap = train_acc - test_acc

    rows.append([
        model,
        round(train_acc, 4),
        round(test_acc, 4),
        round(gap, 4)
    ])

df = pd.DataFrame(rows, columns=[
    "Model", "Train Accuracy", "Test Accuracy", "Generalization Gap"
])

print("\n===== Model Robustness Table =====")
print(df)

# =========================
# 3. 保存CSV（论文用）
# =========================
df.to_csv("results/model_robustness_table.csv", index=False)

# =========================
# 4. 画图（升级版，不再画丑图）
# =========================
plt.figure(figsize=(8, 5))

x = range(len(df["Model"]))

plt.bar(x, df["Train Accuracy"], width=0.3, label="Train Acc")
plt.bar([i + 0.3 for i in x], df["Test Accuracy"], width=0.3, label="Test Acc")

plt.xticks([i + 0.15 for i in x], df["Model"])
plt.ylim(0.85, 1.0)
plt.title("Train vs Test Accuracy Comparison")
plt.legend()

plt.tight_layout()
plt.savefig("results/train_test_comparison.png", dpi=300)
plt.show()

# =========================
# 5. gap可视化（更论文风）
# =========================
plt.figure(figsize=(6, 4))
plt.bar(df["Model"], df["Generalization Gap"], color="tomato")
plt.title("Generalization Gap (Overfitting Indicator)")
plt.ylabel("Train - Test Accuracy")

plt.tight_layout()
plt.savefig("results/generalization_gap.png", dpi=300)
plt.show()