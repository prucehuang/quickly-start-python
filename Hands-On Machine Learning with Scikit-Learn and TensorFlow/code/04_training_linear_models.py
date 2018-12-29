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
from sklearn.linear_model import LinearRegression

X = 2 * np.random.rand(100, 1)
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
y = 4 + 3 * X + np.random.randn(100, 1)

# 正规方程求解
def normal_equation():
    # 方式一，使用numpy函数
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)  # 上公式
    print(theta_best)  # [[3.95336514] [3.07080187]]
    # 方式二，使用sklearn函数
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print(lin_reg.intercept_, lin_reg.coef_)  # [3.95336514] [[3.07080187]]
    # 预测两个点 并将这两个点连成线段画出来
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
    y_predict = X_new_b.dot(theta_best)
    print(y_predict)
    plt.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
    plt.legend(loc="upper left", fontsize=14)
    plt.show()

if __name__ == "__main__":
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.axis([0, 2, 0, 15])

    '''
        用正规方程直接求解最优解
    '''
    # normal_equation()

    '''
        梯度下降
    '''
    eta = 0.1
    n_iterations = 1000
    m = 100
    theta = np.random.randn(2, 1)

    for iteration in range(n_iterations):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients