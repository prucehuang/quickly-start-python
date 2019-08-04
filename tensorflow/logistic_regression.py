# !usr/bin/env python
# coding:utf-8

"""
Implement Logistic Regression with Mini-batch Gradient Descent using TensorFlow.
Train it and evaluate it on the moons dataset.

author: prucehuang 
 email: 1756983926@qq.com
  date: 2019/08/03
"""

import sys, os
for i in range(len(sys.path)):
    sys.path[i] = sys.path[i].split(' ')[0]

import matplotlib.pyplot as plt
import numpy as  np
import tensorflow as tf
from sklearn.datasets import make_moons
from sklearn.metrics import precision_score, recall_score
from datetime import datetime
from scipy.stats import reciprocal

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# 随机batch
def random_batch(X_train, y_train, batch_size):
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]
    y_batch = y_train[rnd_indices]
    return X_batch, y_batch

def plot_data(X, y, is_bias=0):
    plt.plot(X[y == 1, 0+is_bias], X[y == 1, 1+is_bias], 'go', label="Positive")
    plt.plot(X[y == 0, 0+is_bias], X[y == 0, 1+is_bias], 'r^', label="Negative")
    plt.legend()
    plt.show()

"""
    使用GradientDescentOptimizer，跑通tensorflow
"""
def v1():
    # 获得数据， 并show一下
    m = 1000
    X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)
    plot_data(X_moons, y_moons)
    X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]
    y_moons_column_vector = y_moons.reshape(-1, 1)
    print(X_moons.shape, y_moons.shape, X_moons_with_bias.shape,
          y_moons_column_vector.shape)  # (1000, 2) (1000,) (1000, 3) (1000, 1)
    # 将数据切割成 训练数据 和 测试数据
    test_ratio = 0.2
    test_size = int(m * test_ratio)
    X_train = X_moons_with_bias[:-test_size]
    X_test = X_moons_with_bias[-test_size:]
    y_train = y_moons_column_vector[:-test_size]
    y_test = y_moons_column_vector[-test_size:]
    # the construction phase 定义Graph
    reset_graph()
    n_inputs = 2
    X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n_inputs + 1, 1], -1.0, 1.0, seed=42), name="theta")
    logits = tf.matmul(X, theta, name="logits")
    y_proba = tf.sigmoid(logits)
    loss = tf.losses.log_loss(y, y_proba)  # uses epsilon = 1e-7 by default
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    # the execution phase 开始执行
    n_epochs = 1000
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):  # 迭代 n_epochs 次
            for batch_index in range(n_batches):  # 每次走n_batches步
                X_batch, y_batch = random_batch(X_train, y_train, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            loss_val = loss.eval({X: X_test, y: y_test})
            if epoch % 100 == 0:
                print("Epoch:", epoch, "\tLoss:", loss_val)

        y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})

    y_pred = (y_proba_val >= 0.5)
    print('precision score:', precision_score(y_test, y_pred), 'recall score:', recall_score(y_test, y_pred))
    # precision score: 0.8514851485148515 recall score: 0.8686868686868687
    plot_data(X_test, y_pred.reshape(-1), 1)

def logistic_regression(X, y, initializer=None, seed=42, learning_rate=0.01):
    n_inputs_including_bias = int(X.get_shape()[1])
    with tf.name_scope("logistic_regression"):
        with tf.name_scope("model"):
            if initializer is None:
                initializer = tf.random_uniform([n_inputs_including_bias, 1], -1.0, 1.0, seed=seed)
            theta = tf.Variable(initializer, name="theta")
            logits = tf.matmul(X, theta, name="logits")
            y_proba = tf.sigmoid(logits)
        with tf.name_scope("train"):
            loss = tf.losses.log_loss(y, y_proba, scope="loss")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)
            loss_summary = tf.summary.scalar('log_loss', loss)
        with tf.name_scope("init"):
            init = tf.global_variables_initializer()
        with tf.name_scope("save"):
            saver = tf.train.Saver()
    return y_proba, loss, training_op, loss_summary, init, saver

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "D:/tmp/tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

"""
    跑通tensorflow的同时，增加模块化、checkpoint、saver、tensorboard元素
"""
def v2():
    # 获得数据
    m = 1000
    X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)
    X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]
    y_moons_column_vector = y_moons.reshape(-1, 1)
    print(X_moons.shape, y_moons.shape, X_moons_with_bias.shape,
          y_moons_column_vector.shape)  # (1000, 2) (1000,) (1000, 3) (1000, 1)
    # 将数据切割成 训练数据 和 测试数据
    test_ratio = 0.2
    test_size = int(m * test_ratio)
    X_train = X_moons_with_bias[:-test_size]
    X_test = X_moons_with_bias[-test_size:]
    y_train = y_moons_column_vector[:-test_size]
    y_test = y_moons_column_vector[-test_size:]
    X_train_enhanced = np.c_[X_train,
                             np.square(X_train[:, 1]),
                             np.square(X_train[:, 2]),
                             X_train[:, 1] ** 3,
                             X_train[:, 2] ** 3]
    X_test_enhanced = np.c_[X_test,
                            np.square(X_test[:, 1]),
                            np.square(X_test[:, 2]),
                            X_test[:, 1] ** 3,
                            X_test[:, 2] ** 3]
    # the construction phase 定义Graph
    reset_graph()
    n_inputs = 2 + 4
    logdir = log_dir("logreg")

    X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

    y_proba, loss, training_op, loss_summary, init, saver = logistic_regression(X, y)

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    n_epochs = 10001
    batch_size = 50
    n_batches = int(np.ceil(m / batch_size))

    checkpoint_path = "D:/tmp/tf_logs/my_logreg_model.ckpt"
    checkpoint_epoch_path = checkpoint_path + ".epoch"
    final_model_path = "D:/tmp/tf_logs/my_logreg_model"

    with tf.Session() as sess:
        if os.path.isfile(checkpoint_epoch_path):
            # if the checkpoint file exists, restore the model and load the epoch number
            with open(checkpoint_epoch_path, "rb") as f:
                start_epoch = int(f.read())
            print("Training was interrupted. Continuing at epoch", start_epoch)
            saver.restore(sess, checkpoint_path)
        else:
            start_epoch = 0
            sess.run(init)

        for epoch in range(start_epoch, n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = random_batch(X_train_enhanced, y_train, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={X: X_test_enhanced, y: y_test})
            file_writer.add_summary(summary_str, epoch)
            if epoch % 500 == 0:
                print("Epoch:", epoch, "\tLoss:", loss_val)
                saver.save(sess, checkpoint_path)
                with open(checkpoint_epoch_path, "wb") as f:
                    f.write(b"%d" % (epoch + 1))

        saver.save(sess, final_model_path)
        y_proba_val = y_proba.eval(feed_dict={X: X_test_enhanced, y: y_test})
        os.remove(checkpoint_epoch_path)
        y_pred = (y_proba_val >= 0.5)
        print('precision score:', precision_score(y_test, y_pred), 'recall score:', recall_score(y_test, y_pred))
        # precision score: 0.9797979797979798 recall score: 0.9797979797979798
        plot_data(X_test, y_pred.reshape(-1), 1)

""" 
    增加自动调参功能 
"""
def v3():
    # 获得数据
    m = 1000
    X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)
    X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]
    y_moons_column_vector = y_moons.reshape(-1, 1)
    print(X_moons.shape, y_moons.shape, X_moons_with_bias.shape,
          y_moons_column_vector.shape)  # (1000, 2) (1000,) (1000, 3) (1000, 1)
    # 将数据切割成 训练数据 和 测试数据
    test_ratio = 0.2
    test_size = int(m * test_ratio)
    X_train = X_moons_with_bias[:-test_size]
    X_test = X_moons_with_bias[-test_size:]
    y_train = y_moons_column_vector[:-test_size]
    y_test = y_moons_column_vector[-test_size:]
    X_train_enhanced = np.c_[X_train,
                             np.square(X_train[:, 1]),
                             np.square(X_train[:, 2]),
                             X_train[:, 1] ** 3,
                             X_train[:, 2] ** 3]
    X_test_enhanced = np.c_[X_test,
                            np.square(X_test[:, 1]),
                            np.square(X_test[:, 2]),
                            X_test[:, 1] ** 3,
                            X_test[:, 2] ** 3]
    # the construction phase 定义Graph
    reset_graph()
    n_search_iterations = 10

    for search_iteration in range(n_search_iterations):
        batch_size = np.random.randint(1, 100)
        learning_rate = reciprocal(0.0001, 0.1).rvs(random_state=search_iteration)

        n_inputs = 2 + 4
        logdir = log_dir("logreg")

        print("Iteration", search_iteration)
        print("  logdir:", logdir)
        print("  batch size:", batch_size)
        print("  learning_rate:", learning_rate)
        print("  training: ", end="")

        reset_graph()

        X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
        y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

        y_proba, loss, training_op, loss_summary, init, saver = logistic_regression(
            X, y, learning_rate=learning_rate)

        file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

        n_epochs = 10001
        n_batches = int(np.ceil(m / batch_size))

        final_model_path = "D:/tmp/tf_logs/my_logreg_model_%d" % search_iteration

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(n_epochs):
                for batch_index in range(n_batches):
                    X_batch, y_batch = random_batch(X_train_enhanced, y_train, batch_size)
                    sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
                loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={X: X_test_enhanced, y: y_test})
                file_writer.add_summary(summary_str, epoch)
                if epoch % 500 == 0:
                    print(".", end="")

            saver.save(sess, final_model_path)

            print()
            y_proba_val = y_proba.eval(feed_dict={X: X_test_enhanced, y: y_test})
            y_pred = (y_proba_val >= 0.5)

            print("  precision:", precision_score(y_test, y_pred))
            print("  recall:", recall_score(y_test, y_pred))

if __name__ == "__main__":
    v1()
