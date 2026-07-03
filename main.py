from src.preprocess import DataPreprocessor

pre = DataPreprocessor()

X_train, y_train, X_val, y_val, X_test, y_test = pre.preprocess()

print("处理完成")