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
from matplotlib.colors import ListedColormap

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

    per_clf = Perceptron(max_iter=100, random_state=42, tol=1e-3)
    per_clf.fit(X, y)

    y_pred = per_clf.predict([[2, 0.5]])
    print(y_pred, per_clf.coef_, per_clf.intercept_)
    a = -per_clf.coef_[0][0] / per_clf.coef_[0][1]
    b = -per_clf.intercept_ / per_clf.coef_[0][1]

    axes = [0, 5, 0, 2]

    x0, x1 = np.meshgrid(
        np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
        np.linspace(axes[2], axes[3], 200).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = per_clf.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 0, 0], X[y == 0, 1], "bs", label="Not Iris-Setosa")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "yo", label="Iris-Setosa")
    # 画出决策边界
    plt.plot([axes[0], axes[1]], [a * axes[0] + b, a * axes[1] + b], "k-", linewidth=3)
    # 上底色
    custom_cmap = ListedColormap(['#9898ff', '#fafab0'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.axis(axes)

    save_fig("perceptron_iris_plot")
    plt.show()

def logit(z):
    return 1 / (1 + np.exp(-z))

'''the rectified linear unit 修正后的线性单元'''
def relu(z):
    return np.maximum(0, z)

'''函数f在z点上的求导'''
def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)

def plot_activation_functions():
    z = np.linspace(-5, 5, 200)
    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plt.plot(z, np.sign(z), "r-", linewidth=2, label="Step") # 计算数组各元素的符号值
    plt.plot(z, logit(z), "g--", linewidth=2, label="Logit")
    plt.plot(z, np.tanh(z), "b-", linewidth=2, label="Tanh")
    plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
    plt.grid(True)
    plt.legend(loc="center right", fontsize=14)
    plt.title("Activation functions", fontsize=14)
    plt.axis([-5, 5, -1.2, 1.2])

    plt.subplot(122)
    plt.plot(z, derivative(np.sign, z), "r-", linewidth=2, label="Step")
    plt.plot(0, 0, "ro", markersize=5)
    plt.plot(0, 0, "rx", markersize=10)
    plt.plot(z, derivative(logit, z), "g--", linewidth=2, label="Logit")
    plt.plot(z, derivative(np.tanh, z), "b-", linewidth=2, label="Tanh")
    plt.plot(z, derivative(relu, z), "m-.", linewidth=2, label="ReLU")
    plt.grid(True)
    # plt.legend(loc="center right", fontsize=14)
    plt.title("Derivatives", fontsize=14)
    plt.axis([-5, 5, -0.2, 1.2])

    save_fig("activation_functions_plot")
    plt.show()

def heaviside(z):
    return (z >= 0).astype(z.dtype)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def mlp_xor(x1, x2, activation=heaviside):
    return activation(-activation(x1 + x2 - 1.5) + activation(x1 + x2 - 0.5) - 0.5)

'''使用两层结构解决线性不可分数据分类问题'''
def plot_2_layer_network():
    x1s = np.linspace(-0.2, 1.2, 100)
    x2s = np.linspace(-0.2, 1.2, 100)
    x1, x2 = np.meshgrid(x1s, x2s)

    z1 = mlp_xor(x1, x2, activation=heaviside)
    z2 = mlp_xor(x1, x2, activation=sigmoid)

    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.contourf(x1, x2, z1)
    plt.plot([0, 1], [0, 1], "gs", markersize=20)
    plt.plot([0, 1], [1, 0], "y^", markersize=20)
    plt.title("Activation function: heaviside", fontsize=14)
    plt.grid(True)

    plt.subplot(122)
    plt.contourf(x1, x2, z2)
    plt.plot([0, 1], [0, 1], "gs", markersize=20)
    plt.plot([0, 1], [1, 0], "y^", markersize=20)
    plt.title("Activation function: sigmoid", fontsize=14)
    plt.grid(True)

    plt.show()

'''DNNClassifier分类手写字体'''
def train_model_for_mnist_with_estimator_api():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # print(X_train.shape, X_test.shape) # (60000, 28, 28) (10000,)
    X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]
    # 模型训练
    feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
    dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                         feature_columns=feature_cols)
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True)
    dnn_clf.train(input_fn=input_fn)
    # 模型评估
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"X": X_test}, y=y_test, shuffle=False)
    eval_results = dnn_clf.evaluate(input_fn=test_input_fn)
    print(eval_results) # {'accuracy': 0.9799, 'average_loss': 0.09833317, 'loss': 12.447237, 'global_step': 44000}
    # print predict result
    y_pred_iter = dnn_clf.predict(input_fn=test_input_fn)
    print(type(y_pred_iter), list(y_pred_iter)[0])
    '''{'logits': array([ -5.2575245,   1.7127547,  -3.1512573,   5.672271 ,  -4.4269605,
        -9.6272955, -15.6611   ,  26.25397  ,  -1.2106177,   4.8464465],
      dtype=float32), 
     'probabilities': array([2.0641068e-14, 2.1972840e-11, 1.6961831e-13, 1.1520771e-09,
       4.7363277e-14, 2.6119518e-16, 6.2591671e-19, 1.0000000e+00,
       1.1810866e-12, 5.0446525e-10], dtype=float32), 
     'class_ids': array([7], dtype=int64), 
     'classes': array([b'7'], dtype=object)} '''

def train_model_for_mnist_with_plain_api():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # print(X_train.shape, X_test.shape) # (60000, 28, 28) (10000,)
    X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    X_valid, X_train = X_train[:5000], X_train[5000:]
    y_valid, y_train = y_train[:5000], y_train[5000:]

    n_inputs = 28 * 28  # MNIST
    n_hidden1 = 300
    n_hidden2 = 100
    n_outputs = 10
    reset_graph()

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int32, shape=(None), name="y")

    '''创建一层神经网络'''
    def neuron_layer(X, n_neurons, name, activation=None):
        with tf.name_scope(name):
            n_inputs = int(X.get_shape()[1])
            stddev = 2 / np.sqrt(n_inputs)
            init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
            W = tf.Variable(init, name="kernel")
            b = tf.Variable(tf.zeros([n_neurons]), name="bias")
            Z = tf.matmul(X, W) + b
            if activation is not None:
                return activation(Z)
            else:
                return Z

    with tf.name_scope("dnn"):
        hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                               activation=tf.nn.relu)
        hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                               activation=tf.nn.relu)
        logits = neuron_layer(hidden2, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    learning_rate = 0.01

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 40
    batch_size = 50

    def shuffle_batch(X, y, batch_size):
        rnd_idx = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch_idx in np.array_split(rnd_idx, n_batches):
            X_batch, y_batch = X[batch_idx], y[batch_idx]
            yield X_batch, y_batch

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
            print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)

        print(saver.save(sess, "./my_model_final.ckpt"))

    with tf.Session() as sess:
        saver.restore(sess, "./my_model_final.ckpt")  # or better, use save_path
        X_new_scaled = X_test[:20]
        Z = logits.eval(feed_dict={X: X_new_scaled})
        y_pred = np.argmax(Z, axis=1)

    print("Predicted classes:", y_pred)
    print("Actual classes:   ", y_test[:20])

if __name__ == "__main__":
    print('Hello, Welcome to My World')
    # high_level_percetron()
    # plot_activation_functions()
    # plot_2_layer_network()
    # train_model_for_mnist_with_estimator_api()