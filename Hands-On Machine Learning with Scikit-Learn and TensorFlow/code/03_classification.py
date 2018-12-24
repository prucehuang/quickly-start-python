# !usr/bin/env python
# coding:utf-8

"""
分类
author: prucehuang
@email: 1756983926@qq.com
@modify:
@time: 2018/12/18
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, precision_recall_curve, \
    roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_val_predict

DIGIT_IMAGE_SIZE = 8

def plot_digit(data):
    image = data.reshape(DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE)
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")

def plot_digits(instances, images_per_row=10, **options):
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1 # //是向下取整除法，为了最后+1一定有意义，所以len后-1，针对处理整除的情况
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1)) # 列增加，行数不变
    image = np.concatenate(row_images, axis=0) # 行增加，列数不变
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

def show_digit(X, y):
    some_digit = X[360]
    some_digit_lable = y[360]
    print(some_digit_lable)
    plot_digit(some_digit)
    plt.show()

    plt.figure(figsize=(9, 9))
    # X[a:b:c] 从第a+1行开始，到b-1行为止， 每隔c行取一行
    example_images = np.r_[X[:120:6], X[130:306:6], X[300:600:5]]
    print(np.r_[y[:120:6], y[130:306:6], y[300:600:5]].reshape((len(example_images)-1)//10 +1, 10))
    plot_digits(example_images, images_per_row=10)
    plt.show()

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1.2])

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

def print_model_predict_evaluate(y_train, y_train_pred):
    # 交叉矩阵
    # [[1063   14]
    #   [5  118]]
    print('confusion_matrix\n', confusion_matrix(y_train, y_train_pred))
    # 准确率
    print('precision_score:', precision_score(y_train, y_train_pred))
    # 召回率
    print('recall_score:', recall_score(y_train, y_train_pred))
    # f1
    print('f1:', f1_score(y_train, y_train_pred))

if __name__ == "__main__":
    pd.set_option('display.width', 1000)  # 设置字符显示宽度
    pd.set_option('display.max_columns', None)  # 打印所有列，类似的max_rows打印所有行
    # data, target, target_names, images, descr
    mnist = load_digits()
    X, y = mnist["data"], mnist["target"]
    print(X.shape, y.shape)
    '''
        show picture
    '''
    # show_digit(X, y)

    some_digit = X[360]
    some_digit_lable = y[360]
    X_train, X_test, y_train, y_test = X[:1200], X[1200:], y[:1200], y[1200:]
    shuffle_index = np.random.permutation(1200)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    # SVM model
    sgd_clf = SGDClassifier(random_state=42, max_iter=100, tol=1e-3)
    # sgd_clf.fit(X_train, y_train_5)
    # print(sgd_clf.predict([some_digit]), some_digit_lable)
    # print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
    # 一直返回0 用以证明正例比较小的场景
    # never_5_clf = Never5Classifier()
    # print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    print('---------------SGD-------------------')
    print_model_predict_evaluate(y_train_5, y_train_pred)

    # 用交叉验证选择的model预测，结果返回score
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    # 阈值-准确率、召回率曲线图
    # plot_precision_recall_vs_threshold(precisions, recalls, thresholds) # 根据曲线 选择了thresholds=0
    # 召回率-准确率曲线
    # plot_precision_vs_recall(precisions, recalls)
    # roc曲线
    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
    # plot_roc_curve(fpr, tpr)
    # AUC
    print('SGD AUC', roc_auc_score(y_train_5, y_scores))

    forest_clf = RandomForestClassifier(random_state=42, n_estimators=10)
    y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
    print('---------------RandomForestClassifier-------------------')
    print_model_predict_evaluate(y_train_5, y_train_pred_forest)

    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1]  # score = proba of positive class - 结构是[0的概率， 1的概率]
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
    print('RandomForest AUC', roc_auc_score(y_train_5, y_scores_forest))
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
    plt.legend(loc="lower right", fontsize=16)
    plt.show()

    # plt.show()

    
    
    
    
    
    
    
    
    
    
