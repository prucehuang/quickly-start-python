# !usr/bin/env python
# coding:utf-8

"""
分类
author: prucehuang
@email: 1756983926@qq.com
@modify:
@time: 2018/12/18
"""
import pandas as pd
from sklearn.datasets import load_digits
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    pd.set_option('display.width', 1000)  # 设置字符显示宽度
    pd.set_option('display.max_columns', None)  # 打印所有列，类似的max_rows打印所有行
    # data, target, target_names, images, descr
    mnist = load_digits()
    X, y = mnist["data"], mnist["target"]
    print(X.shape, y.shape)
    some_digit = X[30]
    some_digit_image = some_digit.reshape(8, 8)
    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")
    plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
