# !usr/bin/env python
# coding:utf-8

"""

author: prucehuang 
 email: 1756983926@qq.com
  date: 2019/01/19
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(42)

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "unsupervised_learning"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def dimensionality_reducetion_svd_pca():
    '''使用svd函数求解特征向量'''
    X_centered = X - X.mean(axis=0)  # 中心化
    # https://blog.csdn.net/u012162613/article/details/42214205
    # V是一个(n, n)的矩阵，每一列是一个特征方向，Vt每一行是一个特征向量
    # 奇异值分解指的是将矩阵A 分解成三个矩阵相乘(m, n) = (m, m) * (m, n) * (n, n)
    # 即 A = U * Sigma * Vt Sigma除了对角线都是0，且按照由大到小的顺序排列，对角线的值称为奇异值
    # 特征压缩的时候，减少奇异值的个数，(m, n) = (m, d) * (d, d) * (d, n)
    U, s, Vt = np.linalg.svd(X_centered)
    W2 = Vt.T[:, :2]  # 取前两个维度
    print('Top 2 法向量', W2)
    X2D_svd = X_centered.dot(W2)  # (m, 2) = (m, n) * (n, 2)
    print('压缩后的数据', X2D_svd[:5])
    # 逆生长
    X3D_svd_inv = X2D_svd.dot(Vt[:2, :])  # (m, 3) = (m, 2) * (2, 3)
    print('压缩损失', np.mean(np.sum(np.square(X3D_svd_inv - X), axis=1)))

    '''使用PCA就行压缩'''
    pca = PCA(n_components=2)
    X2D_pca = pca.fit_transform(X)
    print('压缩后的数据', X2D_pca[:5])
    # 逆生长
    X3D_inv = pca.inverse_transform(X2D_pca)
    print(np.mean(np.sum(np.square(X3D_inv - X), axis=1)))
    # 成分，即 Vt
    print('components(Vt)', pca.components_)
    print('variance_ratio', pca.explained_variance_ratio_, 'lost variance', 1 - pca.explained_variance_ratio_.sum())

if __name__ == "__main__":
    np.random.seed(4)
    m = 60
    w1, w2 = 0.1, 0.3
    noise = 0.1
    angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
    X = np.empty((m, 3))
    X[:, 0] = np.cos(angles) + np.sin(angles) / 2 + noise * np.random.randn(m) / 2
    X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
    X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

    '''使用svd和pca压缩数据'''
    dimensionality_reducetion_svd_pca()


