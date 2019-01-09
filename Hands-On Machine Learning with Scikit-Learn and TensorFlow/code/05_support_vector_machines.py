# !usr/bin/env python
# coding:utf-8

"""
SVM 实践
author: prucehuang 
 email: 1756983926@qq.com
  date: 2019/01/08
"""
from sklearn.datasets import make_moons
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC, LinearSVC
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0] / w[1] * x0 - b / w[1]

    margin = 1 / w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

def hard_margin_classification():
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]
    setosa_or_versicolor = (y == 0) | (y == 1)
    X = X[setosa_or_versicolor]  # 只选出两个分类的样本
    y = y[setosa_or_versicolor]

    # SVM Classifier model
    svm_clf = SVC(kernel="linear", C=float("inf"))
    svm_clf.fit(X, y)

    # Bad models
    x0 = np.linspace(0, 5.5, 200)
    pred_1 = 5 * x0 - 20
    pred_2 = x0 - 1.8
    pred_3 = 0.1 * x0 + 0.5

    plt.figure(figsize=(12, 2.7))

    plt.subplot(121)
    plt.plot(x0, pred_1, "g--", linewidth=2)
    plt.plot(x0, pred_2, "m-", linewidth=2)
    plt.plot(x0, pred_3, "r-", linewidth=2)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs", label="Iris-Versicolor")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo", label="Iris-Setosa")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    plt.subplot(122)
    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "yo")
    plt.xlabel("Petal length", fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    plt.show()

def sensitivity_to_feature_scales():
    Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
    ys = np.array([0, 0, 1, 1])
    svm_clf = SVC(kernel="linear", C=100)
    svm_clf.fit(Xs, ys)

    plt.figure(figsize=(12, 3.2))
    plt.subplot(121)
    plt.plot(Xs[:, 0][ys == 1], Xs[:, 1][ys == 1], "bo")
    plt.plot(Xs[:, 0][ys == 0], Xs[:, 1][ys == 0], "ms")
    plot_svc_decision_boundary(svm_clf, 0, 6)
    plt.xlabel("$x_0$", fontsize=20)
    plt.ylabel("$x_1$  ", fontsize=20, rotation=0)
    plt.title("Unscaled", fontsize=16)
    plt.axis([0, 6, 0, 90])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xs)
    svm_clf.fit(X_scaled, ys)

    plt.subplot(122)
    plt.plot(X_scaled[:, 0][ys == 1], X_scaled[:, 1][ys == 1], "bo")
    plt.plot(X_scaled[:, 0][ys == 0], X_scaled[:, 1][ys == 0], "ms")
    plot_svc_decision_boundary(svm_clf, -2, 2)
    plt.xlabel("$x_0$", fontsize=20)
    plt.title("Scaled", fontsize=16)
    plt.axis([-2, 2, -2, 2])

    plt.show()

def sensitivity_to_outliers():
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]
    setosa_or_versicolor = (y == 0) | (y == 1)
    X = X[setosa_or_versicolor]  # 只选出两个分类的样本
    y = y[setosa_or_versicolor]
    X_outliers = np.array([[3.4, 1.3], [3.2, 0.8]])
    y_outliers = np.array([0, 0])
    Xo1 = np.concatenate([X, X_outliers[:1]], axis=0)
    yo1 = np.concatenate([y, y_outliers[:1]], axis=0)
    Xo2 = np.concatenate([X, X_outliers[1:]], axis=0)
    yo2 = np.concatenate([y, y_outliers[1:]], axis=0)

    svm_clf = SVC(kernel="linear", C=10 ** 9)
    svm_clf.fit(Xo2, yo2)

    plt.figure(figsize=(12, 2.7))

    plt.subplot(121)
    plt.plot(Xo1[:, 0][yo1 == 1], Xo1[:, 1][yo1 == 1], "bs")
    plt.plot(Xo1[:, 0][yo1 == 0], Xo1[:, 1][yo1 == 0], "yo")
    plt.text(0.3, 1.0, "Impossible!", fontsize=24, color="red")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.annotate("Outlier",
                 xy=(X_outliers[0][0], X_outliers[0][1]),
                 xytext=(2.5, 1.7),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 fontsize=16,
                 )
    plt.axis([0, 5.5, 0, 2])

    plt.subplot(122)
    plt.plot(Xo2[:, 0][yo2 == 1], Xo2[:, 1][yo2 == 1], "bs")
    plt.plot(Xo2[:, 0][yo2 == 0], Xo2[:, 1][yo2 == 0], "yo")
    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.xlabel("Petal length", fontsize=14)
    plt.annotate("Outlier",
                 xy=(X_outliers[1][0], X_outliers[1][1]),
                 xytext=(3.2, 0.08),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 fontsize=16,
                 )
    plt.axis([0, 5.5, 0, 2])

    plt.show()

def three_way_to_linear_svm():
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

    svm_clf_linear_svc = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
    ])

    svm_clf_sgd = Pipeline([
        ("scaler", StandardScaler()),
        ("sgd_clf", SGDClassifier(loss="hinge", alpha=1 / 100, random_state=42, max_iter=1000, tol=1e-3)),
    ])

    svm_clf_kernel = Pipeline([
        ("scaler", StandardScaler()),
        ("svc_kernel", SVC(kernel="linear", C=1, random_state=42)),
    ])

    svm_clf_linear_svc.fit(X, y)
    print(svm_clf_linear_svc.named_steps, 'predict ==>', svm_clf_linear_svc.predict([[5.5, 1.7]]))

    svm_clf_sgd.fit(X, y)
    print(svm_clf_sgd.named_steps, 'predict ==>', svm_clf_sgd.predict([[5.5, 1.7]]))

    svm_clf_kernel.fit(X, y)
    print(svm_clf_kernel.named_steps, 'predict ==>', svm_clf_kernel.predict([[5.5, 1.7]]))

def comparing_different_regularization_c_settings():
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica

    scaler = StandardScaler()
    svm_clf_c_1 = LinearSVC(C=1, loss="hinge", random_state=42)
    scaled_svm_clf_c_1 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf_c_1),
    ])
    scaled_svm_clf_c_1.fit(X, y)

    svm_clf_c_100 = LinearSVC(C=100, loss="hinge", random_state=42)
    scaled_svm_clf_c_100 = Pipeline([
        ("scaler", scaler),
        ("linear_svc", svm_clf_c_100),
    ])
    scaled_svm_clf_c_100.fit(X, y)

    # Convert to unscaled parameters
    b1 = svm_clf_c_1.decision_function([-scaler.mean_ / scaler.scale_])
    b2 = svm_clf_c_100.decision_function([-scaler.mean_ / scaler.scale_])
    w1 = svm_clf_c_1.coef_[0] / scaler.scale_
    w2 = svm_clf_c_100.coef_[0] / scaler.scale_
    svm_clf_c_1.intercept_ = np.array([b1])
    svm_clf_c_100.intercept_ = np.array([b2])
    svm_clf_c_1.coef_ = np.array([w1])
    svm_clf_c_100.coef_ = np.array([w2])

    # Find support vectors (LinearSVC does not do this automatically)
    t = y * 2 - 1
    support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
    support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()
    svm_clf_c_1.support_vectors_ = X[support_vectors_idx1]
    svm_clf_c_100.support_vectors_ = X[support_vectors_idx2]

    plt.figure(figsize=(12, 3.2))
    plt.subplot(121)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^", label="Iris-Virginica")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs", label="Iris-Versicolor")
    plot_svc_decision_boundary(svm_clf_c_1, 4, 6)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.title("$C = {}$".format(svm_clf_c_1.C), fontsize=16)
    plt.axis([4, 6, 0.8, 2.8])

    plt.subplot(122)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plot_svc_decision_boundary(svm_clf_c_100, 4, 6)
    plt.xlabel("Petal length", fontsize=14)
    plt.title("$C = {}$".format(svm_clf_c_100.C), fontsize=16)
    plt.axis([4, 6, 0.8, 2.8])

    plt.show()

# 将一维的非线性可分数据 转换成 二维的线性可分数据
def plot_polynomial_change_linear_data():
    X1D = np.linspace(-4, 4, 9).reshape(-1, 1)
    X2D = np.c_[X1D, X1D ** 2]
    y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plt.grid(True, which='both')  # 显示网格
    plt.axhline(y=0, color='k')  # 画出y=0的横坐标轴
    plt.plot(X1D[:, 0][y == 0], np.zeros(4), "bs")
    plt.plot(X1D[:, 0][y == 1], np.zeros(5), "g^")
    plt.gca().get_yaxis().set_ticks([])  # 重新设置y轴显示的数字刻度为[]
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.axis([-4.5, 4.5, -0.2, 0.2])

    plt.subplot(122)
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.plot(X2D[:, 0][y == 0], X2D[:, 1][y == 0], "bs")
    plt.plot(X2D[:, 0][y == 1], X2D[:, 1][y == 1], "g^")
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
    plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])
    plt.plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)
    plt.axis([-4.5, 4.5, -1, 17])

    plt.subplots_adjust(right=1)
    plt.show()

if __name__ == "__main__":
    '''首先展示一下SVM的预测曲线和支持向量'''
    # hard_margin_classification()
    '''正式的证实特征处理的重要性'''
    # sensitivity_to_feature_scales()
    '''异常值对hard margin classification的影响'''
    # sensitivity_to_outliers()
    '''三种方式实现线性SVM'''
    # three_way_to_linear_svm()
    '''对比超参C对线性SVM的影响，C越大street越窄'''
    # comparing_different_regularization_c_settings()
    '''多项式kernel'''
    # plot_polynomial_change_linear_data()
    X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

    def plot_dataset(X, y, axes):
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
        plt.axis(axes)
        plt.grid(True, which='both')
        plt.xlabel(r"$x_1$", fontsize=20)
        plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    plt.show()
    polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])

    polynomial_svm_clf.fit(X, y)

    def plot_predictions(clf, axes):
        x0s = np.linspace(axes[0], axes[1], 100)
        x1s = np.linspace(axes[2], axes[3], 100)
        x0, x1 = np.meshgrid(x0s, x1s)
        X = np.c_[x0.ravel(), x1.ravel()]
        y_pred = clf.predict(X).reshape(x0.shape)
        y_decision = clf.decision_function(X).reshape(x0.shape)
        plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
        plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

    plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])

    plt.show()