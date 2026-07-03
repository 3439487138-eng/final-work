# Dry Bean Classification Project

本项目基于 Dry Bean 数据集完成多分类机器学习任务，主要比较 KNN、SVM、Logistic Regression 和 Random Forest 四种模型在清洗后数据集上的分类效果。

## 1. 项目简介

本项目首先对原始 Dry Bean 数据进行清洗和预处理，包括缺失值处理、异常值处理、特征标准化和标签编码。随后分别训练 KNN、SVM、Logistic Regression 和 Random Forest 模型，并通过准确率、精确率、召回率、F1 值、混淆矩阵、学习曲线和鲁棒性分析等指标进行综合评价。

## 2. 项目结构

```text
final work/
├── data/            数据文件
├── results/         实验结果、图像和报告
├── src/             主要代码
│   ├── preprocess.py
│   ├── KNN.py
│   ├── SVM.py
│   ├── LR.py
│   ├── RF.py
│   ├── metric_compare.py
│   ├── robustness.py
│   └── learning_curve_all_models.py
└── main.py          主程序入口
## 3. 使用的模型
```

本项目主要使用以下四种机器学习分类模型：

- K-Nearest Neighbors（KNN）
- Support Vector Machine（SVM）
- Logistic Regression（逻辑回归）
- Random Forest（随机森林）

其中，KNN 通过邻近样本投票完成分类；SVM 通过构造最优分类超平面完成分类；逻辑回归作为线性分类模型，用于对比基础模型效果；随机森林通过集成多棵决策树提高分类稳定性。

## 4. 评价指标

为了全面评价模型分类效果，本项目使用了以下指标：

- Accuracy：整体分类准确率
- Precision：精确率
- Recall：召回率
- F1-score：精确率和召回率的综合指标
- Confusion Matrix：混淆矩阵，用于观察各类别的分类情况
- Learning Curve：学习曲线，用于分析模型训练效果和过拟合情况
- Robustness Analysis：鲁棒性分析，用于测试模型在不同数据条件下的稳定性

## 5. 运行方法

运行主程序：

```bash
python main.py
```
## 6. 实验结果

实验结果主要保存在 `results` 文件夹中，包括模型性能对比图、混淆矩阵、学习曲线、训练集与测试集对比图、鲁棒性分析图以及相关报告文件。

主要结果文件包括：

```text
results/
├── four_models_comparison.png        四个模型性能对比图
├── knn_confusion_matrix.png          KNN 混淆矩阵
├── svm_confusion_matrix.png          SVM 混淆矩阵
├── lr_confusion_matrix.png           Logistic Regression 混淆矩阵
├── rf_confusion_matrix.png           Random Forest 混淆矩阵
├── train_test_comparison.png         训练集与测试集性能对比
├── learning_curve_KNN.png            KNN 学习曲线
├── learning_curve_svm.png            SVM 学习曲线
├── learning_curve_LR.png             Logistic Regression 学习曲线
├── learning_curve_RF.png             Random Forest 学习曲线
├── noise_robustness.png              噪声鲁棒性分析
├── data_size_robustness.png          数据规模鲁棒性分析
└── cv_stability.png                  交叉验证稳定性分析
```
从整体实验结果来看，SVM 模型在本项目中的分类表现较好，具有较高的准确率和较稳定的泛化能力。KNN 和 Random Forest 也取得了较好的分类效果，而 Logistic Regression 由于模型表达能力相对有限，在复杂非线性分类任务中的表现相对较弱。

## 7. 项目总结

本项目完成了从数据预处理、模型训练、模型评价到结果分析的完整机器学习分类流程。通过对 KNN、SVM、Logistic Regression 和 Random Forest 四种模型的实验对比，可以发现不同模型在分类效果、泛化能力和鲁棒性方面存在一定差异。

综合准确率、精确率、召回率、F1 值、混淆矩阵、学习曲线和鲁棒性分析结果，SVM 在本任务中表现较为突出，说明其更适合当前 Dry Bean 多分类数据集。KNN 和 Random Forest 的整体表现也较稳定，可以作为有效的对比模型。Logistic Regression 作为线性模型，虽然结构简单、可解释性较强，但在本项目中的分类效果相对不如其他非线性或集成模型。

通过本项目，可以较完整地展示机器学习分类任务的一般流程，包括数据清洗、特征处理、模型训练、性能评价、模型对比以及结果可视化分析。
