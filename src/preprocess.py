import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        self.medians = {}
        self.outlier_bounds = {}

        self.non_negative_cols = [
            "Area",
            "Perimeter",
            "MajorAxisLength",
            "MinorAxisLength",
            "ConvexArea",
            "EquivDiameter",
            "Extent",
            "Solidity",
            "Roundness",
            "Compactness",
            "ShapeFactor1",
            "ShapeFactor2",
            "ShapeFactor3",
            "ShapeFactor4"
        ]

    # =========================
    # 1. 读取数据
    # =========================
    def load_data(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))

        train_path = os.path.join(base_dir, "data/raw/Dry_Bean_Dataset_Dirty_train.csv")
        val_path = os.path.join(base_dir, "data/raw/Dry_Bean_Dataset_Dirty_val.csv")
        test_path = os.path.join(base_dir, "data/raw/Dry_Bean_Dataset_Dirty_test.csv")

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)

        return train_df, val_df, test_df

    # =========================
    # 2. 清洗 Class 标签
    # =========================
    def clean_label(self, df, label_col="Class"):
        df = df.copy()

        def fix_label(x):
            x = str(x).strip().upper()

            # 修正数字混入字母的问题
            x = x.replace("0", "O")
            x = x.replace("3", "E")

            label_map = {
                "BARBUNYA": "BARBUNYA",
                "BOMBAY": "BOMBAY",
                "CALI": "CALI",
                "DERMASON": "DERMASON",
                "HOROZ": "HOROZ",
                "SEKER": "SEKER",
                "SIRA": "SIRA",
            }

            return label_map.get(x, np.nan)

        df[label_col] = df[label_col].apply(fix_label)

        # 如果还有无法识别的标签，直接删除
        before = len(df)
        df = df.dropna(subset=[label_col])
        after = len(df)

        if before != after:
            print(f"删除无法识别标签样本数：{before - after}")

        return df

    # =========================
    # 3. 基础数值清洗
    # =========================
    def basic_clean(self, df):
        df = df.copy()

        # 1. 把 ? 替换成 NaN
        df = df.replace("?", np.nan)

        # 2. 除 Class 外全部转数值
        for col in df.columns:
            if col != "Class":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # =========================
    # 4. 处理不合理负值
    # =========================
    def replace_negative_values(self, df):
        df = df.copy()

        for col in self.non_negative_cols:
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan

        return df

    # =========================
    # 5. 用训练集计算中位数
    # =========================
    def fit_medians(self, train_df):
        self.medians = {}

        for col in train_df.columns:
            if col != "Class":
                self.medians[col] = train_df[col].median()

    # =========================
    # 6. 用训练集中位数填充缺失值
    # =========================
    def fill_missing_with_train_median(self, df):
        df = df.copy()

        for col in df.columns:
            if col != "Class":
                df[col] = df[col].fillna(self.medians[col])

        return df

    # =========================
    # 7. IQR异常值边界
    # =========================
    def fit_outlier_bounds(self, train_df):
        self.outlier_bounds = {}

        for col in train_df.columns:
            if col != "Class":
                q1 = train_df[col].quantile(0.25)
                q3 = train_df[col].quantile(0.75)
                iqr = q3 - q1

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                self.outlier_bounds[col] = (lower, upper)

    # =========================
    # 8. IQR截断异常值
    # =========================
    def clip_outliers(self, df):
        df = df.copy()

        for col, (lower, upper) in self.outlier_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)

        return df

    # =========================
    # 9. 标签编码
    # =========================
    def encode_label(self, train_df, val_df, test_df, label_col="Class"):
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        train_df[label_col] = self.label_encoder.fit_transform(train_df[label_col])
        val_df[label_col] = self.label_encoder.transform(val_df[label_col])
        test_df[label_col] = self.label_encoder.transform(test_df[label_col])

        print("标签编码对应关系：")
        for i, cls in enumerate(self.label_encoder.classes_):
            print(i, "->", cls)

        return train_df, val_df, test_df

    # =========================
    # 10. 标准化
    # =========================
    def scale_features(self, X_train, X_val, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled

    # =========================
    # 11. 保存 processed 数据
    # =========================
    def save_processed(self, X_train, y_train, X_val, y_val, X_test, y_test, feature_cols):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        save_dir = os.path.join(base_dir, "data/processed")

        os.makedirs(save_dir, exist_ok=True)

        train_processed = pd.DataFrame(X_train, columns=feature_cols)
        train_processed["Class"] = y_train.values

        val_processed = pd.DataFrame(X_val, columns=feature_cols)
        val_processed["Class"] = y_val.values

        test_processed = pd.DataFrame(X_test, columns=feature_cols)
        test_processed["Class"] = y_test.values

        train_processed.to_csv(os.path.join(save_dir, "train_processed.csv"), index=False)
        val_processed.to_csv(os.path.join(save_dir, "val_processed.csv"), index=False)
        test_processed.to_csv(os.path.join(save_dir, "test_processed.csv"), index=False)

    # =========================
    # 12. 主流程
    # =========================
    def preprocess(self, save=True, use_iqr=True):
        print("📊 开始数据预处理...")

        train_df, val_df, test_df = self.load_data()

        print("原始数据读取完成")
        print("Train:", train_df.shape)
        print("Val:", val_df.shape)
        print("Test:", test_df.shape)

        # 1. 标签清洗：必须在标签编码前完成
        train_df = self.clean_label(train_df)
        val_df = self.clean_label(val_df)
        test_df = self.clean_label(test_df)

        print("清洗后标签类别数：")
        print("Train:", train_df["Class"].nunique())
        print("Val:", val_df["Class"].nunique())
        print("Test:", test_df["Class"].nunique())

        # 2. 数值清洗
        train_df = self.basic_clean(train_df)
        val_df = self.basic_clean(val_df)
        test_df = self.basic_clean(test_df)

        # 3. 负值异常处理
        train_df = self.replace_negative_values(train_df)
        val_df = self.replace_negative_values(val_df)
        test_df = self.replace_negative_values(test_df)

        # 4. 缺失值填充
        self.fit_medians(train_df)

        train_df = self.fill_missing_with_train_median(train_df)
        val_df = self.fill_missing_with_train_median(val_df)
        test_df = self.fill_missing_with_train_median(test_df)

        # 5. IQR异常值处理
        if use_iqr:
            self.fit_outlier_bounds(train_df)

            train_df = self.clip_outliers(train_df)
            val_df = self.clip_outliers(val_df)
            test_df = self.clip_outliers(test_df)

            print("已完成 IQR 异常值截断处理")

        # 6. 标签编码
        train_df, val_df, test_df = self.encode_label(train_df, val_df, test_df)

        # 7. 划分特征和标签
        X_train = train_df.drop(columns=["Class"])
        y_train = train_df["Class"]

        X_val = val_df.drop(columns=["Class"])
        y_val = val_df["Class"]

        X_test = test_df.drop(columns=["Class"])
        y_test = test_df["Class"]

        feature_cols = X_train.columns

        # 8. 标准化
        X_train, X_val, X_test = self.scale_features(X_train, X_val, X_test)

        # 9. 保存
        if save:
            self.save_processed(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                feature_cols
            )
            print("💾 processed 数据已保存到 data/processed")

        print("✅ 数据预处理完成")

        return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    pre = DataPreprocessor()
    pre.preprocess()