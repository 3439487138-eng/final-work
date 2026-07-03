import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# =========================
# 1. 统一模型定义
# =========================
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", C=1),
    "LR": LogisticRegression(max_iter=1000),
    "RF": RandomForestClassifier(n_estimators=100, random_state=42)
}


# =========================
# 2. 画 learning curve
# =========================
def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.plot(train_sizes, train_mean, marker="o", label="Train Score")
    plt.plot(train_sizes, val_mean, marker="o", label="Validation Score")

    plt.title(f"Learning Curve ({title})")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    plt.ylim(0.85, 1.0)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"results/learning_curve_{title}.png", dpi=300)
    plt.show()


# =========================
# 3. 主函数
# =========================
def main(X, y):
    import os
    os.makedirs("results", exist_ok=True)

    for name, model in models.items():
        print(f"Training learning curve for {name} ...")
        plot_learning_curve(model, X, y, name)


# =========================
# 4. 示例调用（你改成自己的数据）
# =========================
if __name__ == "__main__":
    from preprocess import DataPreprocessor

    dp = DataPreprocessor()
    X_train, y_train, X_val, y_val, X_test, y_test = dp.preprocess()

    # 合并 train + val（学习曲线必须用整体数据）
    import numpy as np
    X = np.vstack([X_train, X_val])
    y = np.hstack([y_train, y_val])

    main(X, y)