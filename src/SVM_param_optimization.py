import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# 1. 路径设置
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "data/processed/train_processed.csv")
VAL_PATH = os.path.join(BASE_DIR, "data/processed/val_processed.csv")
TEST_PATH = os.path.join(BASE_DIR, "data/processed/test_processed.csv")

RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================
# 2. 读取数据
# =========================
def load_data():
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    test = pd.read_csv(TEST_PATH)

    X_train = train.drop(columns=["Class"])
    y_train = train["Class"]

    X_val = val.drop(columns=["Class"])
    y_val = val["Class"]

    X_test = test.drop(columns=["Class"])
    y_test = test["Class"]

    return X_train, y_train, X_val, y_val, X_test, y_test


# =========================
# 3. SVM 参数搜索
# =========================
def svm_parameter_search(X_train, y_train, X_val, y_val):
    kernels = ["linear", "rbf"]
    C_values = [0.1, 0.5, 1, 5, 10]

    results = []

    best_model = None
    best_kernel = None
    best_C = None
    best_acc = 0

    for kernel in kernels:
        for C in C_values:
            print(f"正在训练 SVM: kernel={kernel}, C={C}")

            model = SVC(kernel=kernel, C=C, gamma="scale")
            model.fit(X_train, y_train)

            y_val_pred = model.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)

            results.append({
                "Kernel": kernel,
                "C": C,
                "Validation Accuracy": val_acc
            })

            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model
                best_kernel = kernel
                best_C = C

    results_df = pd.DataFrame(results)

    print("\n========== SVM 参数优化结果 ==========")
    print(results_df)
    print("\n最优参数：")
    print("Best Kernel:", best_kernel)
    print("Best C:", best_C)
    print("Best Validation Accuracy:", best_acc)

    return results_df, best_model, best_kernel, best_C, best_acc


# =========================
# 4. 保存参数搜索结果
# =========================
def save_param_results(results_df):
    save_path = os.path.join(RESULT_DIR, "svm_parameter_search_results.csv")
    results_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print("SVM参数搜索结果已保存到：", save_path)


# =========================
# 5. 绘制参数优化折线图
# =========================
def plot_param_results(results_df):
    plt.figure(figsize=(9, 6))

    for kernel in results_df["Kernel"].unique():
        sub_df = results_df[results_df["Kernel"] == kernel]
        plt.plot(
            sub_df["C"],
            sub_df["Validation Accuracy"],
            marker="o",
            linewidth=2,
            label=f"kernel={kernel}"
        )

        for x, y in zip(sub_df["C"], sub_df["Validation Accuracy"]):
            plt.text(x, y + 0.001, f"{y:.4f}", ha="center", fontsize=9)

    plt.xscale("log")
    plt.xlabel("C Value")
    plt.ylabel("Validation Accuracy")
    plt.title("SVM Parameter Optimization")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(RESULT_DIR, "svm_parameter_optimization.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("SVM参数优化图已保存到：", save_path)


# =========================
# 6. 最优SVM在测试集上评估
# =========================
def evaluate_best_model(best_model, X_test, y_test, best_kernel, best_C):
    y_pred = best_model.predict(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n========== 最优SVM测试集结果 ==========")
    print("Best Kernel:", best_kernel)
    print("Best C:", best_C)
    print("Test Accuracy:", test_acc)
    print(report)

    # 保存测试结果
    report_path = os.path.join(RESULT_DIR, "svm_best_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("========== Best SVM Result ==========\n")
        f.write(f"Best Kernel: {best_kernel}\n")
        f.write(f"Best C: {best_C}\n")
        f.write(f"Test Accuracy: {test_acc}\n\n")
        f.write(report)

    # 保存混淆矩阵CSV
    cm_csv_path = os.path.join(RESULT_DIR, "svm_best_confusion_matrix.csv")
    pd.DataFrame(cm).to_csv(cm_csv_path, index=False)

    # 绘制混淆矩阵
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Best SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()

    cm_png_path = os.path.join(RESULT_DIR, "svm_best_confusion_matrix.png")
    plt.savefig(cm_png_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("最优SVM报告已保存到：", report_path)
    print("最优SVM混淆矩阵图已保存到：", cm_png_path)


# =========================
# 7. 主函数
# =========================
def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    results_df, best_model, best_kernel, best_C, best_acc = svm_parameter_search(
        X_train, y_train, X_val, y_val
    )

    save_param_results(results_df)
    plot_param_results(results_df)

    evaluate_best_model(
        best_model,
        X_test,
        y_test,
        best_kernel,
        best_C
    )


if __name__ == "__main__":
    main()