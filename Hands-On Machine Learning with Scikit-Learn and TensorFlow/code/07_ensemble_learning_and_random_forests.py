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
from sklearn.datasets import make_moons, load_iris
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor

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

# 对比几种集成学习方法的准确率
def ensemble_bagging_pasting():
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 500颗决策树的bagging
    bag_decision_tree_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42), n_estimators=500,
        max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
    bag_decision_tree_clf.fit(X_train, y_train)
    y_pred = bag_decision_tree_clf.predict(X_test)
    print('bag_decision_tree_clf', accuracy_score(y_test, y_pred))  # 0.904
    # 500颗决策树的pasting
    past_decision_tree_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42), n_estimators=500,
        max_samples=100, bootstrap=False, n_jobs=-1, random_state=42)
    past_decision_tree_clf.fit(X_train, y_train)
    y_pred = past_decision_tree_clf.predict(X_test)
    print('past_decision_tree_clf', accuracy_score(y_test, y_pred))  # 0.912
    # 500颗随机决策树的bagging
    bag_rnd_tree_clf = BaggingClassifier(
        DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
        n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)
    bag_rnd_tree_clf.fit(X_train, y_train)
    y_pred = bag_rnd_tree_clf.predict(X_test)
    print('bag_rnd_tree_clf', accuracy_score(y_test, y_pred)) # 0.92
    # 500颗树的随机森林
    rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
    rnd_clf.fit(X_train, y_train)
    y_pred_rf = rnd_clf.predict(X_test)
    print('rnd_clf', accuracy_score(y_test, y_pred)) # 0.92
    # 1颗决策树
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    y_pred_tree = tree_clf.predict(X_test)
    print('tree_clf', accuracy_score(y_test, y_pred_tree))  # 0.856
    export_graphviz(
        tree_clf,
        out_file=image_path("decision_tree.dot"),
        feature_names=["x1", "x2"],
        rounded=True,
        filled=True
    )
    plt.figure(figsize=(12, 6))

    plt.subplot(231)
    plot_decision_boundary(bag_decision_tree_clf, X, y)
    plt.title("Decision Trees with Bagging", fontsize=14)

    plt.subplot(232)
    plot_decision_boundary(past_decision_tree_clf, X, y, contour=True)
    plt.title("Decision Trees with Pasting", fontsize=14)

    plt.subplot(233)
    plot_decision_boundary(bag_rnd_tree_clf, X, y, contour=True)
    plt.title("Random Decision Trees with Bagging", fontsize=14)

    plt.subplot(234)
    plot_decision_boundary(rnd_clf, X, y, contour=True)
    plt.title("Random Decision Trees", fontsize=14)

    plt.subplot(235)
    plot_decision_boundary(tree_clf, X, y, contour=True)
    plt.title("Decision Trees", fontsize=14)

    save_fig("decision_tree_without_and_with_bagging_plot")
    plt.show()

# 决策树可以直接输出特征的权重
def feature_importances_weight():
    iris = load_iris()
    rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
    rnd_clf.fit(iris["data"], iris["target"])
    # sepal length(cm) 0.11249225099876374
    # sepal width(cm) 0.023119288282510326
    # petal length(cm) 0.44103046436395765
    # petal width(cm) 0.4233579963547681
    # 直接输出特征的重要程度
    for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
        print(name, score)

# 使用oob验证获得准确率
def out_of_bag_evaluation():
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    bag_clf = BaggingClassifier(
        DecisionTreeClassifier(random_state=42), n_estimators=500,
        bootstrap=True, n_jobs=-1, oob_score=True, random_state=40)
    bag_clf.fit(X_train, y_train)
    print(bag_clf.oob_score_)

    y_pred = bag_clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))

# adaboost 实操，连续训练模型，加大预测错误的样本权重
def adaboost_classifier():
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    m = len(X_train)
    # 对比权重变化速率的影响，学习率越小，模型越稳定
    plt.figure(figsize=(12, 4))
    for subplot, learning_rate in ((131, 1), (132, 0.5)):
        sample_weights = np.ones(m)
        plt.subplot(subplot)
        for i in range(5):
            svm_clf = SVC(kernel="rbf", C=0.05, random_state=42, gamma='auto')
            svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
            y_pred = svm_clf.predict(X_train)
            sample_weights[y_pred != y_train] *= (1 + learning_rate)
            y_pred_tree = svm_clf.predict(X_test)
            print('SCV learning_rate:', learning_rate, i, accuracy_score(y_test, y_pred_tree))
            plot_decision_boundary(svm_clf, X, y, alpha=0.2)
            plt.title("learning_rate = {}".format(learning_rate), fontsize=16)
        if subplot == 131:
            plt.text(-0.7, -0.65, "1", fontsize=14)
            plt.text(-0.6, -0.10, "2", fontsize=14)
            plt.text(-0.5, 0.10, "3", fontsize=14)
            plt.text(-0.4, 0.55, "4", fontsize=14)
            plt.text(-0.3, 0.90, "5", fontsize=14)

    # 200颗树的AdaBoost模型
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5, random_state=42)  # SAMME.R是soft vote
    ada_clf.fit(X_train, y_train)
    y_pred_tree = ada_clf.predict(X_test)
    print('AdaBoost Decision Tree', accuracy_score(y_test, y_pred_tree))  # 0.896
    plt.subplot(133)
    plot_decision_boundary(ada_clf, X, y)
    plt.title('AdaBoost Decision Tree', fontsize=16)
    save_fig("boosting_plot")
    plt.show()

def plot_gbrt_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=10)
    plt.axis(axes)

def gradient_boosting_regression():
    np.random.seed(42)
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)

    tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg1.fit(X, y)

    y2 = y - tree_reg1.predict(X)
    tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg2.fit(X, y2)

    y3 = y2 - tree_reg2.predict(X)
    tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg3.fit(X, y3)

    # X_new = np.array([[0.8]])
    # y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
    # print(y_pred) # 0.75026781

    plt.figure(figsize=(12, 12))
    plt.subplot(331)
    plot_gbrt_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-",
                     data_label="Training set")
    plt.ylabel("$y$", fontsize=16, rotation=0)
    plt.title("Residuals and tree predictions", fontsize=12)

    plt.subplot(332)
    plot_gbrt_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$",
                     data_label="Training set")
    plt.ylabel("$y$", fontsize=16, rotation=0)
    plt.title("Ensemble predictions", fontsize=16)

    plt.subplot(334)
    plot_gbrt_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+",
                     data_label="Residuals")
    plt.ylabel("$y - h_1(x_1)$", fontsize=16)

    plt.subplot(335)
    plot_gbrt_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
    plt.ylabel("$y$", fontsize=16, rotation=0)

    plt.subplot(337)
    plot_gbrt_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
    plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
    plt.xlabel("$x_1$", fontsize=16)

    plt.subplot(338)
    plot_gbrt_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8],
                     label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$y$", fontsize=16, rotation=0)

    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
    gbrt.fit(X, y)
    # 更小的学习率 更大的决策树颗数
    gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
    gbrt_slow.fit(X, y)

    plt.subplot(333)
    plot_gbrt_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
    plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate, gbrt.n_estimators), fontsize=12)

    plt.subplot(336)
    plot_gbrt_predictions([gbrt_slow], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=12)

    save_fig("gradient_boosting_plot")
    plt.show()

# 两种寻找最优决策树集成学习模型的方法
# 1、先训练，再寻找gbrt.staged_predict(X_val)
# 2、一边训练，一边寻找，增量训练 warm_start=True
def gradient_boosting_regression_find_bst_estimators():
    X = np.random.rand(100, 1) - 0.5
    y = 3 * X[:, 0] ** 2 + 0.05 * np.random.randn(100)
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
    gbrt.fit(X_train, y_train)

    errors = [mean_squared_error(y_val, y_pred)
              for y_pred in gbrt.staged_predict(X_val)]
    bst_n_estimators = np.argmin(errors)

    gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
    gbrt_best.fit(X_train, y_train)

    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.plot(errors, "b.-")
    min_error = np.min(errors)
    print("Best model (%d trees), MSE=%f" % (bst_n_estimators, min_error))
    plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
    plt.plot([0, 120], [min_error, min_error], "k--")
    plt.plot(bst_n_estimators, min_error, "ko")
    plt.text(bst_n_estimators, min_error * 1.2, "Minimum", ha="center", fontsize=14)
    plt.axis([0, 120, 0, 0.01])
    plt.xlabel("Number of trees")
    plt.title("Validation error", fontsize=14)

    plt.subplot(132)
    plot_gbrt_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)

    # 增量训练的模式寻找最优的颗数 warm_start=True
    gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)
    min_val_error = float("inf")
    error_going_up = 0
    for n_estimators in range(1, 120):
        gbrt.n_estimators = n_estimators
        gbrt.fit(X_train, y_train)
        y_pred = gbrt.predict(X_val)
        val_error = mean_squared_error(y_val, y_pred)
        if val_error < min_val_error:
            min_val_error = val_error
            error_going_up = 0
        else:
            error_going_up += 1
            if error_going_up == 5:
                break  # early stopping

    plt.subplot(133)
    plot_gbrt_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
    plt.title("Best model (%d trees, early stopping)" % n_estimators, fontsize=14)
    print("Best model (%d trees, early stopping), MSE=%f" % (n_estimators,min_val_error))

    save_fig("early_stopping_gbrt_plot")
    plt.show()

if __name__ == "__main__":
    '''大数定律'''
    # plot_the_law_of_large_numbers()
    '''投票集成学习'''
    # vote_classifier()
    '''ensemble bagging and pasting'''
    # ensemble_bagging_pasting()
    # feature_importances_weight()
    # out_of_bag_evaluation()
    '''Adaboost'''
    # adaboost_classifier()
    '''Gradient boost'''
    # gradient_boosting_regression()
    gradient_boosting_regression_find_bst_estimators()

