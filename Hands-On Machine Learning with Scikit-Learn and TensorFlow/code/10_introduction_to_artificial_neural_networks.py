# !usr/bin/env python
# coding:utf-8

"""
 ANN
author: prucehuang 
 email: 1756983926@qq.com
  date: 2019/03/18
"""
# Common imports
import numpy as np
import os
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "../"
CHAPTER_ID = "ann"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# 直接调用API
def high_level_percetron():
    iris = load_iris()
    X = iris.data[:, (2, 3)]  # petal length, petal width
    y = (iris.target == 0).astype(np.int)

    per_clf = Perceptron(max_iter=100, random_state=42)
    per_clf.fit(X, y)

    y_pred = per_clf.predict([[2, 0.5]])
    print(y_pred)

if __name__ == "__main__":
    print('Hello, Welcome to My World')
    high_level_percetron()