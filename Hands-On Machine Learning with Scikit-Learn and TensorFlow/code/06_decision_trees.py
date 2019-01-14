# !usr/bin/env python
# coding:utf-8

"""
 决策树实践
author: prucehuang 
 email: 1756983926@qq.com
  date: 2019/01/08
"""
import os

from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, make_moons
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "../"
CHAPTER_ID = "decision_trees"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

# 将决策树画出来
def visual_decision_tree():
    iris = load_iris()
    X = iris.data[:, 2:]  # petal length and width
    y = iris.target

    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf.fit(X, y)
    print(tree_clf.predict_proba([[5, 1.5]]))
    print(tree_clf.predict([[5, 1.5]]))

    # 将决策树保存到dot文件
    # 先下载安装 https://graphviz.gitlab.io/download/
    # dot -Tpng iris_tree.dot -o iris_tree.png可以转成png图片
    export_graphviz(
        tree_clf,
        out_file=image_path("iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

# 修改叶子节点的最少样本数，增加偏差，减少方差
def min_samples_leaf_decision_tree():
    Xm, ym = make_moons(n_samples=100, noise=0.25, random_state=53)

    deep_tree_clf1 = DecisionTreeClassifier(random_state=42)
    deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)
    deep_tree_clf1.fit(Xm, ym)
    deep_tree_clf2.fit(Xm, ym)

    plt.figure(figsize=(11, 4))
    plt.subplot(121)
    plot_decision_boundary(deep_tree_clf1, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
    plt.title("No restrictions", fontsize=16)
    plt.subplot(122)
    plot_decision_boundary(deep_tree_clf2, Xm, ym, axes=[-1.5, 2.5, -1, 1.5], iris=False)
    plt.title("min_samples_leaf = {}".format(deep_tree_clf2.min_samples_leaf), fontsize=14)

    save_fig("min_samples_leaf_plot")
    plt.show()

# 验证决策树对极大值的敏感
def sensitive_to_data_decision_tree():
    iris = load_iris()
    X = iris.data[:, 2:]  # petal length and width
    y = iris.target

    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf.fit(X, y)

    plt.figure(figsize=(16, 4))
    plt.subplot(121)
    plot_decision_boundary(tree_clf, X, y)
    plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
    plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
    plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
    plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
    plt.text(1.40, 1.0, "Depth=0", fontsize=15)
    plt.text(3.2, 1.80, "Depth=1", fontsize=13)
    plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)
    plt.title('Decision Tree')

    # 去除了最大值的
    not_widest_versicolor = (X[:, 1] != 1.8) | (y == 2)
    X_tweaked = X[not_widest_versicolor]
    y_tweaked = y[not_widest_versicolor]

    tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)
    tree_clf_tweaked.fit(X_tweaked, y_tweaked)
    plt.subplot(122)
    plot_decision_boundary(tree_clf_tweaked, X_tweaked, y_tweaked, legend=False)
    plt.plot([0, 7.5], [0.8, 0.8], "k-", linewidth=2)
    plt.plot([0, 7.5], [1.75, 1.75], "k--", linewidth=2)
    plt.text(1.0, 0.9, "Depth=0", fontsize=15)
    plt.text(1.0, 1.80, "Depth=1", fontsize=13)
    plt.title('not_widest_versicolor')

    save_fig("decision_tree_instability_plot")
    plt.show()

def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")

def regression_decision_trees():
    # Quadratic training set + noise
    np.random.seed(42)
    m = 200
    X = np.random.rand(m, 1)
    y = 4 * (X - 0.5) ** 2 + np.random.randn(m, 1) / 10

    tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
    tree_reg1.fit(X, y)

    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plot_regression_predictions(tree_reg1, X, y)
    for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
        plt.plot([split, split], [-0.2, 1], style, linewidth=2)
    plt.text(0.21, 0.65, "Depth=0", fontsize=15)
    plt.text(0.01, 0.2, "Depth=1", fontsize=13)
    plt.text(0.65, 0.8, "Depth=1", fontsize=13)
    plt.legend(loc="upper center", fontsize=18)
    plt.title("max_depth=2", fontsize=14)
    export_graphviz(
        tree_reg1,
        out_file=image_path("regression_tree.dot"),
        feature_names=["x1"],
        rounded=True,
        filled=True
    )

    plt.subplot(222)
    tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
    tree_reg2.fit(X, y)
    plot_regression_predictions(tree_reg2, X, y, ylabel=None)
    for split, style in ((0.1973, "k-"), (0.0917, "k--"), (0.7718, "k--")):
        plt.plot([split, split], [-0.2, 1], style, linewidth=2)
    for split in (0.0458, 0.1298, 0.2873, 0.9040):
        plt.plot([split, split], [-0.2, 1], "k:", linewidth=1)
    plt.text(0.3, 0.5, "Depth=2", fontsize=13)
    plt.title("max_depth=3", fontsize=14)

    plt.subplot(223)
    tree_reg3 = DecisionTreeRegressor(random_state=42, max_depth=10)
    tree_reg3.fit(X, y)
    plot_regression_predictions(tree_reg3, X, y, ylabel=None)
    plt.text(0.3, 0.5, "Depth=10", fontsize=13)
    plt.title("max_depth=10", fontsize=14)

    plt.subplot(224)
    tree_reg4 = DecisionTreeRegressor(random_state=42, max_depth=10, min_samples_leaf=10)
    tree_reg4.fit(X, y)
    plot_regression_predictions(tree_reg4, X, y, ylabel=None)
    plt.text(0.3, 0.5, "Depth=10", fontsize=13)
    plt.title("max_depth=10, min_samples_leaf=10", fontsize=14)

    save_fig("tree_regression_plot")
    plt.show()

if __name__ == "__main__":
    '''画出决策树'''
    # visual_decision_tree()
    '''决策树调参'''
    # min_samples_leaf_decision_tree()
    '''对数据敏感的决策树'''
    # sensitive_to_data_decision_tree()
    '''回归决策树'''
    regression_decision_trees()
