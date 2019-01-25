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

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

import matplotlib.pyplot as plt
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
    print(result)
    sess.close()
    # 第二种写法，自动close session
    with tf.Session() as sess:
        x.initializer.run()
        y.initializer.run()
        result = f.eval()  # equivalent to tf.get_default_session().run(f)
        print(result)
    # 第三种写法，统一初始化，自动close
    init = tf.global_variables_initializer()  # prepare an init node
    with tf.Session() as sess:
        init.run()  # actually initialize all the variables
        result = f.eval()
        print(result)
    # 第四种写法，自动初始化，没有with语法段，但需要手动close
    sess = tf.InteractiveSession()
    init.run()
    result = f.eval()
    print(result)
    sess.close()

if __name__ == "__main__":
    print('Hello, Welcome to My World')
    x1 = tf.Variable(1) # default graph
    print(x1.graph is tf.get_default_graph())
    graph = tf.Graph() # 创建一个新的graph
    with graph.as_default():
        x2 = tf.Variable(2)
    print(x2.graph is graph)
    print(x2.graph is tf.get_default_graph())
