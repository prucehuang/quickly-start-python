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
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

np.random.seed(42)

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "../"
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

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_origin_data_and_pca_data():
    axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]

    x1s = np.linspace(axes[0], axes[1], 10)
    x2s = np.linspace(axes[2], axes[3], 10)
    x1, x2 = np.meshgrid(x1s, x2s)

    pca = PCA(n_components=2)
    X2D_pca = pca.fit_transform(X)
    X3D_inv = pca.inverse_transform(X2D_pca)
    C = pca.components_
    R = C.T.dot(C)
    z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])

    fig = plt.figure(figsize=(6, 3.8))
    ax = fig.add_subplot(111, projection='3d')

    X3D_above = X[X[:, 2] > X3D_inv[:, 2]]
    X3D_below = X[X[:, 2] <= X3D_inv[:, 2]]

    ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)

    ax.plot_surface(x1, x2, z, alpha=0.2, color="k")
    np.linalg.norm(C, axis=0)
    ax.add_artist(
        Arrow3D([0, C[0, 0]], [0, C[0, 1]], [0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
    ax.add_artist(
        Arrow3D([0, C[1, 0]], [0, C[1, 1]], [0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
    ax.plot([0], [0], [0], "k.")

    for i in range(m):
        if X[i, 2] > X3D_inv[i, 2]:
            ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-")
        else:
            ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k-", color="#505050")

    ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k+")
    ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k.")
    ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])

    save_fig("dataset_3d_plot")
    plt.show()

    # 压缩后的数据
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    ax.plot(X2D_pca[:, 0], X2D_pca[:, 1], "k+")
    ax.plot(X2D_pca[:, 0], X2D_pca[:, 1], "k.")
    ax.plot([0], [0], "ko")
    ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.set_xlabel("$z_1$", fontsize=18)
    ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
    ax.axis([-1.5, 1.3, -1.2, 1.2])
    ax.grid(True)
    save_fig("dataset_2d_plot")
    plt.show()

# 画出几种场景下的swiss_roll和压缩图
def plot_swiss_roll():
    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
    axes = [-11.5, 14, -2, 23, -12, 15]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
    ax.view_init(10, -70)
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    ax.set_xlim(axes[0:2])  # [-11.5, 14]
    ax.set_ylim(axes[2:4])  # [-2, 23]
    ax.set_zlim(axes[4:6])  # [-12, 15]

    save_fig("swiss_roll_plot")
    plt.show()

    # 直接拍平swiss roll
    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.hot)
    plt.axis(axes[:4])
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18, rotation=0)
    plt.grid(True)

    plt.subplot(122)
    plt.scatter(t, X[:, 1], c=t, cmap=plt.cm.hot)
    plt.axis([4, 15, axes[2], axes[3]])
    plt.xlabel("$z_1$", fontsize=18)
    plt.grid(True)

    save_fig("squished_swiss_roll_plot")
    plt.show()

    axes = [-11.5, 14, -2, 23, -12, 15]

    x2s = np.linspace(axes[2], axes[3], 10)
    x3s = np.linspace(axes[4], axes[5], 10)
    x2, x3 = np.meshgrid(x2s, x3s)

    fig = plt.figure(figsize=(6, 5))
    ax = plt.subplot(111, projection='3d')

    positive_class = X[:, 0] > 5
    X_pos = X[positive_class]
    X_neg = X[~positive_class]
    ax.view_init(10, -70)
    ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
    ax.plot_wireframe(5, x2, x3, alpha=0.5)
    ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])

    save_fig("manifold_decision_boundary_plot1")
    plt.show()

    fig = plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)
    # 高纬的决策边界更简单，具体要根据数据来定
    plt.plot(t[positive_class], X[positive_class, 1], "gs")
    plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
    plt.axis([4, 15, axes[2], axes[3]])
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

    save_fig("manifold_decision_boundary_plot2")
    plt.show()

    fig = plt.figure(figsize=(6, 5))
    ax = plt.subplot(111, projection='3d')

    positive_class = 2 * (t[:] - 4) > X[:, 1]
    X_pos = X[positive_class]
    X_neg = X[~positive_class]
    ax.view_init(10, -70)
    ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
    ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
    ax.set_xlabel("$x_1$", fontsize=18)
    ax.set_ylabel("$x_2$", fontsize=18)
    ax.set_zlabel("$x_3$", fontsize=18)
    ax.set_xlim(axes[0:2])
    ax.set_ylim(axes[2:4])
    ax.set_zlim(axes[4:6])

    save_fig("manifold_decision_boundary_plot3")
    plt.show()

    fig = plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)

    plt.plot(t[positive_class], X[positive_class, 1], "gs")
    plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
    plt.plot([4, 15], [0, 22], "b-", linewidth=2)
    plt.axis([4, 15, axes[2], axes[3]])
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

    save_fig("manifold_decision_boundary_plot4")
    plt.show()

# 简述PCA轴选的不同映射出来的方差差别很大
def plot_pca_axis():
    angle = np.pi / 5
    stretch = 5
    m = 200

    np.random.seed(3)
    X = np.random.randn(m, 2) / 10
    X = X.dot(np.array([[stretch, 0], [0, 1]]))  # stretch
    X = X.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])  # rotate

    u1 = np.array([np.cos(angle), np.sin(angle)])
    u2 = np.array([np.cos(angle - 2 * np.pi / 6), np.sin(angle - 2 * np.pi / 6)])
    u3 = np.array([np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)])

    X_proj1 = X.dot(u1.reshape(-1, 1))
    X_proj2 = X.dot(u2.reshape(-1, 1))
    X_proj3 = X.dot(u3.reshape(-1, 1))

    plt.figure(figsize=(8, 4))
    # plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    plt.subplot(121)
    plt.plot([-1.4, 1.4], [-1.4 * u1[1] / u1[0], 1.4 * u1[1] / u1[0]], "k-", linewidth=1)
    plt.plot([-1.4, 1.4], [-1.4 * u2[1] / u2[0], 1.4 * u2[1] / u2[0]], "k--", linewidth=1)
    plt.plot([-1.4, 1.4], [-1.4 * u3[1] / u3[0], 1.4 * u3[1] / u3[0]], "k:", linewidth=2)
    plt.plot(X[:, 0], X[:, 1], "bo", alpha=0.5)
    plt.axis([-1.4, 1.4, -1.4, 1.4])
    plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k',
              ec='k')
    plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc='k',
              ec='k')
    plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{c_1}$", fontsize=22)
    plt.text(u3[0] + 0.1, u3[1], r"$\mathbf{c_2}$", fontsize=22)
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$x_2$", fontsize=18, rotation=0)
    plt.grid(True)

    # plt.subplot2grid((3, 2), (0, 1))
    plt.subplot(322)
    plt.plot([-2, 2], [0, 0], "k-", linewidth=1)
    plt.plot(X_proj1[:, 0], np.zeros(m), "bo", alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().get_xaxis().set_ticklabels([])
    plt.axis([-2, 2, -1, 1])
    plt.grid(True)

    # plt.subplot2grid((3, 2), (1, 1))
    plt.subplot(324)
    plt.plot([-2, 2], [0, 0], "k--", linewidth=1)
    plt.plot(X_proj2[:, 0], np.zeros(m), "bo", alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().get_xaxis().set_ticklabels([])
    plt.axis([-2, 2, -1, 1])
    plt.grid(True)

    # plt.subplot2grid((3, 2), (2, 1))
    plt.subplot(326)
    plt.plot([-2, 2], [0, 0], "k:", linewidth=2)
    plt.plot(X_proj3[:, 0], np.zeros(m), "bo", alpha=0.3)
    plt.gca().get_yaxis().set_ticks([])
    plt.axis([-2, 2, -1, 1])
    plt.xlabel("$z_1$", fontsize=18)
    plt.grid(True)

    save_fig("pca_best_projection")
    plt.show()

# 对比不同核函数的pca效果
def plot_kpca():
    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

    lin_pca = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=True)
    rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
    sig_pca = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)
    y = t > 6.9
    plt.figure(figsize=(11, 4))
    for subplot, pca, title in ((131, lin_pca, "Linear kernel"),
                                (132, rbf_pca, "RBF kernel, $\gamma=0.04$"),
                                (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
        X_reduced = pca.fit_transform(X)
        if subplot == 132:
            X_reduced_rbf = X_reduced

        plt.subplot(subplot)
        # plt.plot(X_reduced[y, 0], X_reduced[y, 1], "gs")
        # plt.plot(X_reduced[~y, 0], X_reduced[~y, 1], "y^")
        plt.title(title, fontsize=14)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
        plt.xlabel("$z_1$", fontsize=18)
        if subplot == 131:
            plt.ylabel("$z_2$", fontsize=18, rotation=0)
        plt.grid(True)

    save_fig("kernel_pca_plot")
    plt.show()

    # 逆过程压缩
    plt.figure(figsize=(6, 5))
    X_inverse = rbf_pca.inverse_transform(X_reduced_rbf)
    ax = plt.subplot(121, projection='3d')
    ax.view_init(10, -70)
    ax.scatter(X_inverse[:, 0], X_inverse[:, 1], X_inverse[:, 2], c=t, cmap=plt.cm.hot, marker="x")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    save_fig("preimage_plot", tight_layout=False)
    plt.show()

# 为KPCA寻找合适的算法和gamma
def looking_for_param_by_grid_search():
    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
    y = t > 6.9

    clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
    ])

    param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid"]
    }]

    grid_search = GridSearchCV(clf, param_grid, cv=3)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
    X_reduced = rbf_pca.fit_transform(X)
    X_preimage = rbf_pca.inverse_transform(X_reduced)
    print(mean_squared_error(X, X_preimage))

def plot_lle():
    X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
    X_reduced = lle.fit_transform(X)
    plt.title("Unrolled swiss roll using LLE", fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    plt.ylabel("$z_2$", fontsize=18)
    plt.axis([-0.065, 0.055, -0.1, 0.12])
    plt.grid(True)

    save_fig("lle_unrolling_plot")
    plt.show()

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
    # dimensionality_reducetion_svd_pca()
    # plot_origin_data_and_pca_data()
    '''manifold learning'''
    # plot_swiss_roll()
    '''PCA'''
    # plot_pca_axis()
    # plot_kpca()
    '''LLE'''
    # plot_lle()