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
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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

if __name__ == "__main__":
    '''大数定律'''
    # plot_the_law_of_large_numbers()
    '''投票集成学习'''
    # vote_classifier()









