# !usr/bin/env python
# coding:utf-8

"""
分类 - load_digits 手写字体的识别
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
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

DIGIT_IMAGE_SIZE = 8

# 打印单个数字
def plot_digit(data):
    image = data.reshape(DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE)
    plt.imshow(image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")

# 打印多个数字，默认一行最多10个
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

# 稍微展示一下数据
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

# 一个永远返回0的分类器
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass # pass用的妙
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

# 评估曲线 - 横坐标是阈值，纵坐标是准确率、召回率，一般都取相交点
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1.2])
    plt.show()

# 评估曲线 - 横坐标是阈值，纵坐标是F1，选最大值
def plot_f1_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, 1/((1/recalls[:-1])+(1/precisions[:-1])), "g-", label="F1", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 0.8])
    plt.show()

# 评估曲线 - 横坐标是召回率，纵坐标是准确率，根据想要的准确率择取阈值
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.show()

# roc评估曲线 - 横坐标是FPR，纵坐标是召回率TPR，根据想要的召回率选择阈值，或者分析对比不同算法的AUC
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    # plt.show()

# 简单打印一下model的交叉矩阵、准确率、召回率、F1
def print_model_predict_evaluate(model, X_train, y_train_5):
    y_train_pred = cross_val_predict(model, X_train, y_train_5, cv=3)
    # 交叉矩阵
    # [[1063   14]
    #   [5  118]]
    print('confusion_matrix\n', confusion_matrix(y_train_5, y_train_pred))
    # 准确率
    print('precision_score:', precision_score(y_train_5, y_train_pred))
    # 召回率
    print('recall_score:', recall_score(y_train_5, y_train_pred))
    # f1
    print('f1:', f1_score(y_train_5, y_train_pred))

# 简单展示一下各种曲线
def plot_model_predict_curve(model, X_train, y_train_5, method='decision_function'):
    # 用交叉验证选择的model预测，结果返回score
    if method == "decision_function":
        y_scores = cross_val_predict(model, X_train, y_train_5, cv=3, method="decision_function")
    elif method == "predict_proba":
        # score = proba of positive class - 结构是[0的概率， 1的概率]
        y_scores = cross_val_predict(model, X_train, y_train_5, cv=3, method="predict_proba")[:, 1]
    else:
        print('please input correct method value')
        exit(1)
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    print('thresholds count:', len(thresholds))
    # 阈值-准确率、召回率曲线图
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds) # 根据曲线 选择了thresholds=0
    # 阈值-f1曲线
    plot_f1_vs_threshold(precisions, recalls, thresholds)
    # 召回率-准确率曲线
    plot_precision_vs_recall(precisions, recalls)
    # roc曲线
    fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
    plot_roc_curve(fpr, tpr)
    plt.show()
    # AUC
    print('SGD AUC', roc_auc_score(y_train_5, y_scores))

# 二分类算法对比实践
def binary_model_versus(X_train, y_train_5):
    # SVM model
    print('---------------SGD-------------------')
    sgd_clf = SGDClassifier(random_state=42, max_iter=100, tol=1e-3)
    # sgd_clf.fit(X_train, y_train_5)
    # print(sgd_clf.predict([some_digit]), some_digit_lable)
    # SGD模型 VS 非1模型的准确率
    # print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))
    # print(cross_val_score(Never5Classifier(), X_train, y_train_5, cv=3, scoring="accuracy"))
    print_model_predict_evaluate(sgd_clf, X_train, y_train_5)
    plot_model_predict_curve(sgd_clf, X_train, y_train_5)

    print('---------------RandomForestClassifier-------------------')
    forest_clf = RandomForestClassifier(random_state=42, n_estimators=10)
    print_model_predict_evaluate(forest_clf, X_train, y_train_5)
    plot_model_predict_curve(forest_clf, X_train, y_train_5, "predict_proba")

    print('---------------versus model roc curve-------------------')
    y_scores_sgd = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_train_5, y_scores_sgd)

    y_scores_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")[:, 1]
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

    print('SGD AUC', roc_auc_score(y_train_5, y_scores_sgd))
    print('RandomForest AUC', roc_auc_score(y_train_5, y_scores_forest))
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_sgd, tpr_sgd, "b:", linewidth=2, label="SGD")
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
    plt.legend(loc="lower right", fontsize=16)
    plt.show()

def plot_confusion_matrix(matrix):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)

# 多分类器
def multiclass_classification():
    ## 二分类的分类器 —— 线性分类器、SVM
    # 默认是SVM， 默认也是OvA, 训练是十个分类器，选择max的score
    sgd_clf = SGDClassifier(random_state=42, max_iter=100, tol=1e-3)
    sgd_clf.fit(X_train, y_train)
    print('OvA所有的类别：', sgd_clf.classes_)
    print(sgd_clf.predict([some_digit]), '每个分类的概率值：', sgd_clf.decision_function([some_digit]))

    # 强制设定为OVO，一共会生成n*(n-1)/2个分类器
    ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=100, random_state=42, tol=1e-3))
    ovo_clf.fit(X_train, y_train)
    print('OVO所有的类别：', ovo_clf.classes_, '分类器总数：', len(ovo_clf.estimators_))
    print(ovo_clf.predict([some_digit]), '每个分类的概率值：', ovo_clf.decision_function([some_digit]))

    ## 多分类的分类器 —— 随机森林、贝叶斯
    forest_clf = RandomForestClassifier(random_state=42, n_estimators=10)
    forest_clf.fit(X_train, y_train)
    print('随机森林预测：', forest_clf.predict([some_digit]), '每个分类的概率值：', forest_clf.predict_proba([some_digit]))

    # 稍微加上正规化处理一下特征我们的准确率就涨了
    print('特征处理之前的准确率：', cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))
    X_train_scaled = StandardScaler().fit_transform(X_train.astype(np.float64))
    print('特征处理之后的准确率：', cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

    # 多分类的交叉矩阵
    y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)
    print(conf_mx)
    # plt.matshow(conf_mx, cmap=plt.cm.gray)
    # plt.show()
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums
    np.fill_diagonal(norm_conf_mx, 0)  # 将主对角线都设置成0 预测出错的数据就被凸显了
    print(norm_conf_mx)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()
    # 将预测出错的数据单独拉出来分析
    cl_a, cl_b = 1, 8
    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)] # TT
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)] # TF
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)] # FT
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)] # FF

    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plot_digits(X_bb[:25], images_per_row=5)
    plt.subplot(222)
    plot_digits(X_ba[:25], images_per_row=5)
    plt.subplot(223)
    plot_digits(X_ab[:25], images_per_row=5)
    plt.subplot(224)
    plot_digits(X_aa[:25], images_per_row=5)
    plt.show()

# 多label的二分类 分类器
def multilabel_classification():
    y_train_large = (y_train >= 6)
    y_train_odd = (y_train % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd] # 将label横向扩展成两列

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)
    print(knn_clf.predict([some_digit]))
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3, n_jobs=-1)
    print(f1_score(y_multilabel, y_train_knn_pred, average="macro"))

# 多输出分类器
def multioutput_classification():
    noise = np.random.randint(0, 10, (len(X_train), 64))
    X_train_mod = X_train + noise
    noise = np.random.randint(0, 10, (len(X_test), 64))
    X_test_mod = X_test + noise
    y_train_mod = X_train  # 预测的结果已经不再是单个分类，而是每个像素点的颜色值
    y_test_mod = X_test
    # 画出数字
    some_index = 55
    plt.subplot(121);
    plot_digit(X_train_mod[some_index])
    plt.subplot(122);
    plot_digit(y_train_mod[some_index])
    plt.show()

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train_mod, y_train_mod)  # 64个label，每个label有255个取值
    clean_digit = knn_clf.predict([X_test_mod[some_index]])
    plot_digit(clean_digit)
    plt.show()

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

    '''
        get data
    '''
    some_digit = X[360]
    some_digit_lable = y[360]
    X_train, X_test, y_train, y_test = X[:1200], X[1200:], y[:1200], y[1200:]
    shuffle_index = np.random.permutation(1200)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    '''
        model versus
    '''
    binary_model_versus(X_train, y_train_5)

    '''
        Multiclass classification
    '''
    # multiclass_classification()

    '''
        Multilabel classification
    '''
    # multilabel_classification()

    '''
        Multioutput classification
    '''
    # multioutput_classification()




