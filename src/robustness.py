import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================
# 1. 加载数据
# =========================
def load_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data/processed")

    train = np.array(
        __import__("pandas")
        .read_csv(os.path.join(data_dir, "train_processed.csv"))
    )

    val = np.array(
        __import__("pandas")
        .read_csv(os.path.join(data_dir, "val_processed.csv"))
    )

    test = np.array(
        __import__("pandas")
        .read_csv(os.path.join(data_dir, "test_processed.csv"))
    )

    X_train, y_train = train[:, :-1], train[:, -1]
    X_val, y_val = val[:, :-1], val[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    return X_train, y_train, X_test, y_test


# =========================
# 2. 模型初始化
# =========================
def get_models():
    return {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", C=1),
        "LR": LogisticRegression(max_iter=2000),
        "RF": RandomForestClassifier(n_estimators=100)
    }


# =========================
# 3. 噪声鲁棒性
# =========================
def noise_robustness():
    X_train, y_train, X_test, y_test = load_data()
    models = get_models()

    noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]

    results = {name: [] for name in models}

    for sigma in noise_levels:

        X_noisy = X_test + np.random.normal(0, sigma, X_test.shape)

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_noisy)
            acc = accuracy_score(y_test, pred)
            results[name].append(acc)

    # =========================
    # 画图
    # =========================
    plt.figure(figsize=(8, 5))

    for name in models:
        plt.plot(noise_levels, results[name], marker='o', label=name)

    plt.title("Noise Robustness Comparison")
    plt.xlabel("Noise Level (sigma)")
    plt.ylabel("Accuracy")
    plt.ylim(0.89, 0.94)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/noise_robustness.png", dpi=300)
    plt.show()


# =========================
# 4. 数据规模鲁棒性
# =========================
def data_size_robustness():
    X_train, y_train, X_test, y_test = load_data()
    models = get_models()

    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]

    results = {name: [] for name in models}

    n = len(X_train)

    for r in ratios:
        idx = int(n * r)

        X_sub = X_train[:idx]
        y_sub = y_train[:idx]

        for name, model in models.items():
            model.fit(X_sub, y_sub)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            results[name].append(acc)

    # =========================
    # 画图
    # =========================
    plt.figure(figsize=(8, 5))

    for name in models:
        plt.plot(ratios, results[name], marker='o', label=name)

    plt.title("Data Size Robustness Comparison")
    plt.xlabel("Training Data Ratio")
    plt.ylabel("Accuracy")
    plt.ylim(0.9, 0.94)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/data_size_robustness.png", dpi=300)
    plt.show()


# =========================
# 5. 主函数
# =========================
def main():
    noise_robustness()
    data_size_robustness()


if __name__ == "__main__":
    main()