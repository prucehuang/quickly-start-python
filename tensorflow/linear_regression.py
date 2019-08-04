# !usr/bin/env python
# coding:utf-8

"""
Hello World Tensorflow
 真的是方法千万种
author: prucehuang 
 email: 1756983926@qq.com
  date: 2019/08/04
"""

import sys, os
for i in range(len(sys.path)):
    sys.path[i] = sys.path[i].split(' ')[0]

import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# 方法一：用正规方程方法计算theta
def equation_tensorflow():
    X = tf.constant(iris_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(iris.target.reshape(-1, 1), dtype=tf.float32, name="y")
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
    with tf.Session() as sess:
        theta_value = theta.eval()
    print('tensorflow:\n', theta_value)

# 方法二：使用numpy的函数正规方程发求解theta
def equstion_numpy():
    X = iris_data_plus_bias
    y = iris.target.reshape(-1, 1)
    theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    print('numpy:\n', theta_numpy)

# 方法三：使用sklearn的LR class求解
def linear_regression_sklearn():
    lin_reg = LinearRegression()
    lin_reg.fit(iris.data, iris.target.reshape(-1, 1))
    print('sklearn:\n', np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])

# 方法四：使用TensorFlow手动梯度下降
def gradients_descent_manually_tensorflow():
    reset_graph()
    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_iris_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(iris.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    gradients = 2 / m * tf.matmul(tf.transpose(X), error)
    training_op = tf.assign(theta, theta - learning_rate * gradients)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)

        best_theta = theta.eval()
    print('tensorflow manually gradients\n', best_theta)

# 方法五：使用TensorFlow autodiff梯度下降
def gradients_descent_autodiff_tensorflow():
    reset_graph()
    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_iris_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(iris.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    gradients = tf.gradients(mse, [theta])[0]
    training_op = tf.assign(theta, theta - learning_rate * gradients)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)

        best_theta_1 = theta.eval()

    print('tensorflow audodiff gradients:\n', best_theta_1)

# 方法六：直接使用GradientDescentOptimizer
def gradients_descent_optinizer_tensorflow():
    reset_graph()
    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_iris_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(iris.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)
    # start execution
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)
        best_theta = theta.eval()

    print("tensorflow gradient descent optimizer:\n", best_theta)

# 方法七：使用MomentumOptimizer
def gradients_descent_MomentumOptimizer_tnesorflow():
    reset_graph()
    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_iris_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(iris.target.reshape(-1, 1), dtype=tf.float32, name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    # 使用MomentumOptimizer
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    training_op = optimizer.minimize(mse)
    # start execution
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            sess.run(training_op)
            if epoch % 100 == 0:
                print("Epoch", epoch, "MSE =", mse.eval())
        best_theta = theta.eval()

    print("tensorflow momentum optimizer:\n", best_theta)


def fetch_batch(epoch, batch_index, batch_size, n_batches):
    np.random.seed(epoch * n_batches + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_iris_data_plus_bias[indices]
    y_batch = iris.target.reshape(-1, 1)[indices]
    return X_batch, y_batch

# 方法八：随机梯度下降，使用placeholder
def batch_gradients_descent_optinizer_placeholder_tensorflow():
    reset_graph()
    learning_rate = 0.01
    iris = datasets.load_iris()
    m, n = iris.data.shape
    n_epochs = 10
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))
    scaler = StandardScaler()
    scaled_iris_data = scaler.fit_transform(iris.data)
    scaled_iris_data_plus_bias = np.c_[np.ones((m, 1)), scaled_iris_data]
    # placeholder是一个占位符，后续将使用feed_dict往里面添加实际内容
    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
    y_pred = tf.matmul(X, theta, name="predictions")
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size, n_batches)
                # 每次只喂一个批次的随机数据
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        best_theta_2 = theta.eval()

    print("tensorflow mini-batch Gradient Descent Optimizer:\n", best_theta_2)

if __name__ == "__main__":
    print('Hello, Welcome to My World')
    reset_graph()

    iris = datasets.load_iris()
    m, n = iris.data.shape
    iris_data_plus_bias = np.c_[np.ones((m, 1)), iris.data]
    # 特征压缩
    scaler = StandardScaler()
    scaled_iris_data = scaler.fit_transform(iris.data)
    scaled_iris_data_plus_bias = np.c_[np.ones((m, 1)), scaled_iris_data]

    # equation_tensorflow()
    # equstion_numpy()
    # linear_regression_sklearn()
    # gradients_descent_manually_tensorflow()
    # gradients_descent_autodiff_tensorflow()
    # gradients_descent_optinizer_tensorflow()
    # gradients_descent_MomentumOptimizer_tnesorflow()
    batch_gradients_descent_optinizer_placeholder_tensorflow()