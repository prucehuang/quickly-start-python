# !usr/bin/env python
# coding:utf-8

"""

author: prucehuang 
 email: 1756983926@qq.com
  date: 2019/01/14
"""

# Common imports
import numpy as np
import os
import matplotlib.pyplot as plt

# to make this notebook's output stable across runs
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz

np.random.seed(42)

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "../"
CHAPTER_ID = "ensembles"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)

# 展示一下概率
def plot_the_law_of_large_numbers():
    heads_proba = 0.51
    # 创建一个(10000, 10)的01数据
    coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
    # 先按照列梯形累加 再除以当前行号
    cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)

    plt.figure(figsize=(8, 3.5))
    plt.plot(cumulative_heads_ratio)
    plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
    plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
    plt.xlabel("Number of coin tosses")
    plt.ylabel("Heads ratio")
    plt.legend(loc="lower right")
    plt.axis([0, 10000, 0.42, 0.58])
    save_fig("law_of_large_numbers_plot")
    plt.show()

# 全样本，不同算法组合的投票集成学习
# vote比单个算法要好，soft比较hard vote要好
def vote_classifier():
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    log_clf = LogisticRegression(random_state=42, solver='lbfgs')
    rnd_clf = RandomForestClassifier(random_state=42, n_estimators=100)
    svm_clf = SVC(probability=True, random_state=42, gamma='auto', )

    voting_clf_hard = VotingClassifier(
        estimators=[('lr', log_clf),
                    ('rf', rnd_clf),
                    ('svc', svm_clf)],
        voting='hard')

    voting_clf_soft = VotingClassifier(
        estimators=[('lr', log_clf),
                    ('rf', rnd_clf),
                    ('svc', svm_clf)],
        voting='soft')

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf_hard, voting_clf_soft):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

if __name__ == "__main__":
    '''大数定律'''
    # plot_the_law_of_large_numbers()
    '''投票集成学习'''
    # vote_classifier()
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 500颗决策树的bagging
    bag_decision_tree_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
    bag_decision_tree_clf.fit(X_train, y_train)
    y_pred = bag_decision_tree_clf.predict(X_test)
    print('bag_decision_tree_clf', accuracy_score(y_test, y_pred)) #0.904
    # 500颗决策树的pasting
    past_decision_tree_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42), n_estimators=500,
        max_samples=100, bootstrap=False, n_jobs=-1, random_state=42)
    past_decision_tree_clf.fit(X_train, y_train)
    y_pred = past_decision_tree_clf.predict(X_test)
    print('past_decision_tree_clf', accuracy_score(y_test, y_pred)) #0.904
    # 500颗随机决策树的bagging
    bag_rnd_tree_clf = BaggingClassifier(
        DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
        n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)
    bag_rnd_tree_clf.fit(X_train, y_train)
    y_pred = bag_rnd_tree_clf.predict(X_test)
    print('bag_rnd_tree_clf', accuracy_score(y_test, y_pred))
    # 500颗树的随机森林
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
    rnd_clf.fit(X_train, y_train)
    y_pred_rf = rnd_clf.predict(X_test)
    print('rnd_clf', accuracy_score(y_test, y_pred))

    # 1颗决策树
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_test)
    print('tree_clf', accuracy_score(y_test, y_pred_tree)) #0.856
    export_graphviz(
        tree_clf,
        out_file=image_path("decision_tree.dot"),
        feature_names=["x1", "x2"],
        rounded=True,
        filled=True
    )
    plt.figure(figsize=(12,6))

    plt.subplot(231)
    plot_decision_boundary(bag_decision_tree_clf, X, y)
    plt.title("bag_decision_tree_clf", fontsize=14)

    plt.subplot(232)
    plot_decision_boundary(past_decision_tree_clf, X, y, contour=True)
    plt.title("past_decision_tree_clf", fontsize=14)

    plt.subplot(233)
    plot_decision_boundary(bag_rnd_tree_clf, X, y, contour=True)
    plt.title("bag_rnd_tree_clf", fontsize=14)

    plt.subplot(234)
    plot_decision_boundary(rnd_clf, X, y, contour=True)
    plt.title("rnd_clf", fontsize=14)

    plt.subplot(235)
    plot_decision_boundary(tree_clf, X, y, contour=True)
    plt.title("tree_clf", fontsize=14)

    save_fig("decision_tree_without_and_with_bagging_plot")
    plt.show()



