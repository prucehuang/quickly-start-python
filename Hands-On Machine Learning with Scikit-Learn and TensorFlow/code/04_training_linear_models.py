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
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

X = 2 * np.random.rand(100, 1)
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance
m = len(X_b)
y = 1 + 2 * X + np.random.randn(100, 1)

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

# 对比分析三种梯度下降
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

# 画出模型预测的曲线，对比加上正则项的效果
def plot_model(X, y, X_new, model_class, polynomial, alphas, **model_kargs):
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
        lw = 3 if alpha > 0 else 2
        plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha = {}$".format(alpha))
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.axis([0, 3, 0, 4])

# 对比正则项对模型的效果
def regularized_models():
    m = 50
    X = 3 * np.random.rand(m, 1)
    y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
    X_new = np.linspace(0, 3, 100).reshape(100, 1)

    plt.figure(figsize=(16, 8))
    # Ridge Regularized
    plt.subplot(221)
    plot_model(X, y, X_new, Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.title('Ridge')
    plt.subplot(222)
    plot_model(X, y, X_new, Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)
    plt.title('Ridge Polynomial')
    # Lasso Regularized
    plt.subplot(223)
    plot_model(X, y, X_new, Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
    plt.ylabel("$y$", rotation=0, fontsize=18)
    plt.xlabel("$x_1$", fontsize=18)
    plt.title('Lasso')
    plt.subplot(224)
    plot_model(X, y, X_new, Lasso, polynomial=True, alphas=(0, 10**-5, 1), tol=1, random_state=42)
    plt.title('Lasso Polynomial')
    plt.xlabel("$x_1$", fontsize=18)
    plt.show()

    '''
        # L2惩罚项实现
        # 方式一
        ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42) 
        ridge_reg.fit(X, y)
        ridge_reg.predict([[1.5]])
        # 方式二
        sgd_reg = SGDRegressor(max_iter=5, penalty="l2", random_state=42)
        sgd_reg.fit(X, y.ravel())
        sgd_reg.predict([[1.5]])
        # 方式三
        ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
        ridge_reg.fit(X, y)
        ridge_reg.predict([[1.5]])
        
        # L1惩罚项实现
        lasso_reg = Lasso(alpha=0.1)
        lasso_reg.fit(X, y)
        lasso_reg.predict([[1.5]])
        
        # 弹性网络惩罚项实现 
        elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        elastic_net.fit(X, y)
        elastic_net.predict([[1.5]])
    '''

# 用控制迭代次数的办法来控制模型选择参数，选择验证集误差减少后即将增加的拐点时的模型（避免模型进一步训练后过拟合）
def early_stopping():
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)

    X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)

    poly_scaler = Pipeline([
            ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
            ("std_scaler", StandardScaler()),
        ])

    X_train_poly_scaled = poly_scaler.fit_transform(X_train)
    X_val_poly_scaled = poly_scaler.transform(X_val)

    sgd_reg = SGDRegressor(max_iter=2, # 用这个参数可以控制每轮训练几次，外配合总训练轮数就可以打印训练epoch-RMSE曲线
                           tol=1e-3,
                           penalty=None,
                           eta0=0.0005,
                           warm_start=True,
                           learning_rate="constant",
                           random_state=42)

    n_epochs = 500
    train_errors, val_errors = [], []
    # 训练n_epochs，记录训练、验证集误差
    for epoch in range(n_epochs):
        sgd_reg.fit(X_train_poly_scaled, y_train)
        y_train_predict = sgd_reg.predict(X_train_poly_scaled)
        y_val_predict = sgd_reg.predict(X_val_poly_scaled)
        train_errors.append(mean_squared_error(y_train, y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    best_epoch = np.argmin(val_errors) # 返回最小值对应的数组下标
    best_val_rmse = np.sqrt(val_errors[int(best_epoch)])

    plt.annotate('Best model',
                 xy=(best_epoch, best_val_rmse),
                 xytext=(best_epoch, best_val_rmse + 1),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=16,
                )

    best_val_rmse -= 0.03  # just to make the graph look better
    plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
    plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)
    plt.show()

# 画sigmoid曲线
def plot_sigmoid():
    t = np.linspace(-10, 10, 100)
    sigmoid = 1 / (1 + np.exp(-t))
    plt.figure(figsize=(9, 3))
    plt.plot([-10, 10], [0, 0], "k-")
    plt.plot([-10, 10], [0.5, 0.5], "k:")
    plt.plot([-10, 10], [1, 1], "k:")
    plt.plot([0, 0], [-0.1, 1.1], "k-")
    plt.plot(t, sigmoid, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
    plt.xlabel("t")
    plt.legend(loc="upper left", fontsize=18)
    plt.axis([-10, 10, -0.1, 1.1])
    plt.show()

def logistic_regression_binary_classifer():
    iris = datasets.load_iris()
    X = iris["data"][:, 3:]  # petal width
    y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

    log_reg = LogisticRegression(solver='lbfgs', random_state=42)
    log_reg.fit(X, y)

    X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
    y_proba = log_reg.predict_proba(X_new)
    decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

    plt.figure(figsize=(8, 3))
    plt.plot(X[y==0], y[y==0], "bs")
    plt.plot(X[y==1], y[y==1], "g^")
    plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
    plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
    plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
    plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
    plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
    plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
    plt.xlabel("Petal width (cm)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 3, -0.02, 1.02])
    plt.show()

def logistic_regression_multiclass_classifer():
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = (iris["target"] == 2).astype(np.int)  # 选出target==2，Iris-Virginica的分类，并将True|False转为1|0

    log_reg = LogisticRegression(solver='lbfgs', C=10 ** 10, random_state=42)  # C Inverse of regularization strength;
    log_reg.fit(X, y)

    x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),  # x轴
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),  # y轴
    )  # 返回值 x0表示表格每个点的x坐标，x1表示表格每一个点的纵坐标
    X_new = np.c_[x0.ravel(), x1.ravel()]  # 将x0, x1组装成[x, y]

    y_proba = log_reg.predict_proba(X_new)

    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 0, 0], X[y == 0, 1], "bs")  # bs要分开解释，b表示blue，s表示square 所依据bs就是蓝色的正方形
    plt.plot(X[y == 1, 0], X[y == 1, 1], "gh")  # gh是绿色的圆形

    zz = y_proba[:, 1].reshape(x0.shape)
    contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)

    left_right = np.array([2.9, 7])
    boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

    plt.clabel(contour, inline=1, fontsize=12)
    plt.plot(left_right, boundary, "k--", linewidth=3)
    plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
    plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.axis([2.9, 7, 0.8, 2.7])
    plt.show()

def logistic_regression_softmax_classifer():
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]  # petal length, petal width
    y = iris["target"]

    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, random_state=42)
    softmax_reg.fit(X, y)
    # 把全场每个点都预测一遍
    x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    # x0是x坐标 x1是y坐标 y_predict是当前点的分类 y_proba是当前被分类的可信概率
    y_proba = softmax_reg.predict_proba(X_new)
    y_predict = softmax_reg.predict(X_new)

    zz1 = y_proba[:, 1].reshape(x0.shape)  # 用于画出分类=1的等高线
    zz = y_predict.reshape(x0.shape)

    # 将花描点
    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")
    plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")

    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)  # 背景色
    contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)  # 等高线
    plt.clabel(contour, inline=1, fontsize=12)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="center left", fontsize=14)
    plt.axis([0, 7, 0, 3.5])
    plt.show()

# 用逻辑回归处理二分类、多分类问题，用Softmax处理多分类问题
def logistic_regression():
    logistic_regression_binary_classifer()
    logistic_regression_multiclass_classifer()
    logistic_regression_softmax_classifer()

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
    # regularized_models()

    '''遇到拐点就停止训练'''
    # early_stopping()

    '''logistic regression'''
    # plot_sigmoid()
    logistic_regression()
