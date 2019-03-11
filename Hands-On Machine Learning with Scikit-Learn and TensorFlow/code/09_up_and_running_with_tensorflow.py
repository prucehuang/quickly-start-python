# !usr/bin/env python
# coding:utf-8

"""

author: prucehuang 
 email: 1756983926@qq.com
  date: 2019/01/24
"""
# Common imports
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "../"
CHAPTER_ID = "tensorflow"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

'''
    TensorFlow Hello World
'''
def create_graph_and_run_in_session():
    x = tf.Variable(3, name="x")
    y = tf.Variable(4, name="y")
    f = x * x * y + y + 2
    # by now, no computation, just creates a computation graph
    # 第一种写法，手动创建并close
    sess = tf.Session()
    sess.run(x.initializer)
    sess.run(y.initializer)
    result = sess.run(f)
    print('way 1.', result)
    sess.close()
    # 第二种写法，自动close session
    with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        result = f.eval()  # equivalent to tf.get_default_session().run(f)
        print('way 2.', result)
    # 第三种写法，统一初始化，自动close
    init = tf.global_variables_initializer()  # prepare an init node
    with tf.Session() as sess:
        init.run()  # actually initialize all the variables
        result = f.eval()
        print('way 3.', result)
    # 第四种写法，自动初始化，没有with语法段，但需要手动close
    sess = tf.InteractiveSession()
    init.run()
    result = f.eval()
    print('way 4.', result)
    sess.close()

'''
    graph manage
'''
def graph_manage():
    x1 = tf.Variable(1)  # default graph
    print('x1.graph is tf.get_default_graph()', x1.graph is tf.get_default_graph())
    tf.reset_default_graph()
    print('After reset graph, x1.graph is tf.get_default_graph()', x1.graph is tf.get_default_graph())
    graph = tf.Graph()  # 创建一个新的graph
    with graph.as_default():
        x2 = tf.Variable(2)
    print('x2 belong tp a new graph', x2.graph is tf.get_default_graph(), x2.graph is graph)

if __name__ == "__main__":
    # run一个graph
    # create_graph_and_run_in_session()
    # graph管理
    # graph_manage()
    reset_graph()
    iris = datasets.load_iris()
    m, n = iris.data.shape
    iris_data_plus_bias = np.c_[np.ones((m, 1)), iris.data]

    # 方法一：用正规方程方法计算theta
    X = tf.constant(iris_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(iris.target.reshape(-1, 1), dtype=tf.float32, name="y")
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
    with tf.Session() as sess:
        theta_value = theta.eval()
    print('tensorflow:\n', theta_value)

    # 方法二：使用numpy的函数正规方程发求解theta
    X = iris_data_plus_bias
    y = iris.target.reshape(-1, 1)
    theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    print('numpy:\n', theta_numpy)

    # 方法三：使用sklearn的LR class求解
    lin_reg = LinearRegression()
    lin_reg.fit(iris.data, iris.target.reshape(-1, 1))
    print('sklearn:\n', np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])

    # 特征压缩
    scaler = StandardScaler()
    scaled_housing_data = scaler.fit_transform(iris.data)
    scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

    # 方法四：使用TensorFlow手动梯度下降
    reset_graph()
    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
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
    reset_graph()
    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
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
