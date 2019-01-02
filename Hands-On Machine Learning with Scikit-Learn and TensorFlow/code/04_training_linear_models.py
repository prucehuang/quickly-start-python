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
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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

# 批量梯度下降实现
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
    plt.axis([0, 2, 0, 15])
    plt.title(r"BGD $\eta = {}$".format(eta), fontsize=12)

def learning_schedule(t):
    t0, t1 = 100, 2000
    return t0 / (t + t1)

# 随机梯度下降实现，绘制第一轮的前十条预测曲线
def plot_stochastic_gradient_descent(theta, eta_init=0.1, n_iterations=50, theta_path=None):
    plt.plot(X, y, "b.")
    for iteration in range(n_iterations):
        for i in range(m):
            if i==0:
                y_predict = X_new_b.dot(theta)
                style = "b-" if iteration>0 else "r--"
                plt.plot(X_new, y_predict, style)
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = eta_init*m/(iteration * m + i + m) # 定义一个衰减的学习率, 第一轮eta=eta_init
            theta = theta - eta * gradients
            if theta_path is not None:
                theta_path.append(theta)
    plt.axis([0, 2, 0, 15])
    plt.title(r"SGD $\eta = {}$".format(eta_init), fontsize=12)

# 小批量梯度下降
def plot_mine_batch_gradient_descent(theta, eta_init, n_iterations=80, minibatch_size=20, theta_path=None):
    plt.plot(X, y, "b.")
    t = 0
    for iteration in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, minibatch_size):
            if i==0:
                y_predict = X_new_b.dot(theta)
                style = "b-" if iteration > 0 else "r--"
                plt.plot(X_new, y_predict, style)
            xi = X_b_shuffled[i:i + minibatch_size]
            yi = y_shuffled[i:i + minibatch_size]
            gradients = 2 / minibatch_size * xi.T.dot(xi.dot(theta) - yi)
            eta = 100 * eta_init / (t + 1000)  # 定义一个衰减的学习率, 第一轮eta=eta_init
            t += 1
            theta = theta - eta * gradients
            if theta_path is not None:
                theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"MBGD $\eta = {}$".format(eta_init), fontsize=12)

def gradient_descent():
    plt.figure(figsize=(14, 12))
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

    # 绘制theta的改变路径
    theta_path_bgd = np.array(theta_path_bgd)
    theta_path_sgd = np.array(theta_path_sgd)
    theta_path_mbgd = np.array(theta_path_mbgd)
    plt.figure(figsize=(7, 4))
    plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
    plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
    plt.plot(theta_path_mbgd[:, 0], theta_path_mbgd[:, 1], "g-+", linewidth=2, label="Mini-batch")

    plt.legend(loc="upper left", fontsize=16)
    plt.xlabel(r"$\theta_0$", fontsize=20)
    plt.ylabel(r"$\theta_1$   ", fontsize=20, rotation=0)
    plt.axis([2.5, 4.5, 2.3, 3.9])
    plt.show()

# 多项式回归， 使用Pipe来对比不同复杂度模型的表现
def polynomial_regression():
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    X_new = np.linspace(-3, 3, 100).reshape(100, 1)
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

    for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
        polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
        std_scaler = StandardScaler()
        lin_reg = LinearRegression()
        polynomial_regression = Pipeline([
            ("poly_features", polybig_features),  # x0, a, a**2
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ])
        polynomial_regression.fit(X, y)
        y_newbig = polynomial_regression.predict(X_new)
        plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

    plt.plot(X, y, "b.", linewidth=3)
    plt.xlabel("$x_1$", fontsize=18)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([-3, 3, 0, 10])
    plt.show()

# 画出模型随着样本量变化的训练、验证误差
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown

# 通过学习曲线判断模型状态
def learning_curves():
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    lin_reg = LinearRegression()
    plot_learning_curves(lin_reg, X, y)
    plt.axis([0, 80, 0, 3])
    plt.title(r"underfitting_learning_curves_plot", fontsize=12)

    plt.subplot(122)
    std_scaler = StandardScaler()
    polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("std_scaler", std_scaler),
        ("lin_reg", LinearRegression()),
    ])
    plot_learning_curves(polynomial_regression, X, y)
    plt.axis([0, 80, 0, 3])
    plt.title(r"overfitting_learning_curves_plot", fontsize=12)
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)
    '''用正规方程直接求解最优解线性回归'''
    # normal_equation()

    '''梯度下降'''
    # gradient_descent()

    '''多项式回归'''
    # polynomial_regression()

    '''通过学习曲线来判断模型是否过拟合、是否欠拟合、现在模型是什么状态、下一步应该如何处理'''
    # learning_curves()

    '''正则项模型惩罚'''
    m = 20
    X = 3 * np.random.rand(m, 1)
    y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
    X_new = np.linspace(0, 3, 100).reshape(100, 1)


    def plot_model(model_class, polynomial, alphas, **model_kargs):
        for alpha, style in zip(alphas, ("b-", "g--", "r:")):
            model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
            if polynomial:
                model = Pipeline([
                    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                    ("std_scaler", StandardScaler()),
                    ("regul_reg", model),
                ])
            model.fit(X, y)
            y_new_regul = model.predict(X_new)
            lw = 2 if alpha > 0 else 1
            plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
        plt.plot(X, y, "b.", linewidth=3)
        plt.legend(loc="upper left", fontsize=15)
        plt.xlabel("$x_1$", fontsize=18)
        plt.axis([0, 3, 0, 4])

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.subplot(122)
    plot_model(Ridge, polynomial=True, alphas=(0, 10 ** -5, 1), random_state=42)
    plt.show()

