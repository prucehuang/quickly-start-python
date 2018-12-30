# !usr/bin/env python
# coding:utf-8

"""
训练线性模型
author: prucehuang 
 email: 1756983926@qq.com
  date: 2018/12/27
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor

X = 2 * np.random.rand(100, 1)
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
m = len(X_b)
y = 4 + 3 * X + np.random.randn(100, 1)

# 正规方程求解
def normal_equation():
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])
    # 方式一，使用numpy函数
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # 上公式
    print(theta_best)  # [[3.95336514] [3.07080187]]
    # 方式二，使用sklearn函数
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(lin_reg.intercept_, lin_reg.coef_)  # [3.95336514] [[3.07080187]]
    # 预测两个点 并将这两个点连成线段画出来
    y_predict = X_new_b.dot(theta_best)
    print(y_predict)
    plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
    plt.legend(loc="upper left", fontsize=14)
    plt.show()

def plot_batch_gradient_descent(theta, eta, n_iterations=50, theta_path=None):
    plt.plot(X, y, "b.")
    for iteration in range(n_iterations):
        if iteration < 10: # 只画前几次迭代的曲线
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"BGD $\eta = {}$".format(eta), fontsize=16)

def learning_schedule(t):
    return t0 / (t + t1)

def plot_stochastic_gradient_descent(theta, eta_init=0.1, n_iterations=50, theta_path=None):
    plt.plot(X, y, "b.")
    for iteration in range(n_iterations):
        for i in range(m):
            if iteration==0 and i<10:
                y_predict = X_new_b.dot(theta)
                style = "b-" if i > 0 else "r--"
                plt.plot(X_new, y_predict, style)
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = eta_init*m/(iteration * m + i + m) # 定义一个衰减的学习率, 第一轮eta=eta_init
            theta = theta - eta * gradients
            if theta_path is not None:
                theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"SGD $\eta = {}$".format(eta_init), fontsize=16)

def plot_mine_batch_gradient_descent(theta, eta_init, n_iterations=50, minibatch_size=20, theta_path=None):
    plt.plot(X, y, "b.")
    for iteration in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, minibatch_size):
            if iteration==30 and i<10:
                y_predict = X_new_b.dot(theta)
                style = "b-" if i > 0 else "r--"
                plt.plot(X_new, y_predict, style)
            xi = X_b_shuffled[i:i + minibatch_size]
            yi = y_shuffled[i:i + minibatch_size]
            gradients = 2 / minibatch_size * xi.T.dot(xi.dot(theta) - yi)
            eta = eta_init * m / (i + m)  # 定义一个衰减的学习率, 第一轮eta=eta_init
            theta = theta - eta * gradients
            if theta_path is not None:
                theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"MBGD $\eta = {}$".format(eta_init), fontsize=16)

if __name__ == "__main__":
    np.random.seed(42)
    '''
        用正规方程直接求解最优解
    '''
    # normal_equation()

    '''
        梯度下降
    '''
    plt.figure(figsize=(13, 10))
    eta = 0.1
    theta = np.random.randn(2, 1)


    # 绘制不同的学习率给训练带来的影响图
    # Batch Gradient Descent 批量梯度下降
    theta_path_bgd = []
    theta_bgd = theta.copy()
    plt.subplot(331)
    plot_batch_gradient_descent(theta_bgd, eta=0.02)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(332)
    plot_batch_gradient_descent(theta_bgd, eta=0.1, theta_path=theta_path_bgd)
    plt.subplot(333)
    plot_batch_gradient_descent(theta_bgd, eta=0.5)

    # Stochastic Gradient Descent 随机梯度下降
    theta_path_sgd = []
    theta_sgd = theta.copy()
    plt.subplot(334)
    plot_stochastic_gradient_descent(theta_sgd, eta_init=0.02)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(335)
    plot_stochastic_gradient_descent(theta_sgd, eta_init=0.1, theta_path=theta_path_sgd)
    plt.subplot(336)
    plot_stochastic_gradient_descent(theta_sgd, eta_init=0.5)
    # 直接求解随机梯度下降
    # sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42, tol=1e-3)
    # sgd_reg.fit(X, y.ravel())
    # print(sgd_reg.intercept_, sgd_reg.coef_)

    # Mini-batch Gradient Descent 小批量梯度下降
    theta_path_mbgd = []
    theta_mbgd = theta.copy()
    plt.subplot(337)
    plot_mine_batch_gradient_descent(theta_mbgd, eta_init=0.02)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(338)
    plot_mine_batch_gradient_descent(theta_mbgd, eta_init=0.1, theta_path=theta_path_mbgd)
    plt.subplot(339)
    plot_mine_batch_gradient_descent(theta_mbgd, eta_init=0.5)


    # theta_path_bgd = np.array(theta_path_bgd)
    # theta_path_sgd = np.array(theta_path_sgd)
    # theta_path_mgd = np.array(theta_path_mgd)
    # plt.figure(figsize=(7, 4))
    # plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
    # plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
    # plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
    # plt.legend(loc="upper left", fontsize=16)
    # plt.xlabel(r"$\theta_0$", fontsize=20)
    # plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
    # plt.axis([2.5, 4.5, 2.3, 3.9])
    plt.show()
