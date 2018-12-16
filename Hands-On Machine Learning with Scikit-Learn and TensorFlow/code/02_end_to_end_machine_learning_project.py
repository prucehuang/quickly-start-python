# !usr/bin/env python
# coding:utf-8

"""
房价预测
author: prucehuang
@email: 1756983926@qq.com
@modify:
@time: 2018/12/15
"""
import numpy as np
import os, sys
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer

# file path
PROJECT_ROOT_DIR = sys.path[0] + '/../'
HOUSING_PATH = os.path.join(PROJECT_ROOT_DIR, 'datasets', 'housing')

def load_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def quick_look_data():
    print('--------------------------------------------------------------')
    print('housing data head')
    print('----------------------------')
    print(housing.head())

    print('--------------------------------------------------------------')
    print('housing data describe')
    print('----------------------------')
    print(housing.describe())

    print('--------------------------------------------------------------')
    print('housing data info')
    print('----------------------------')
    print(housing.info())
    # longitude             20640 non-null float64
    # latitude              20640 non-null float64
    # housing_median_age    20640 non-null float64 有异常高的值
    # total_rooms           20640 non-null float64
    # total_bedrooms        20433 non-null float64 有空值
    # population            20640 non-null float64
    # households            20640 non-null float64
    # median_income         20640 non-null float64 分段处理，可以用来层次划分训练集测试集
    # median_house_value    20640 non-null float64 有异常高的值
    # ocean_proximity       20640 non-null object  字符串类型的类别
    # 一共九个特征，一共目标房价

    print('--------------------------------------------------------------')
    print('housing data hist')
    print('----------------------------')
    # 将每一个特征用直方图的形式打印出来
    housing.hist(bins=50, figsize=(20, 15))
    plot.show()

    print('--------------------------------------------------------------')
    print('ocean proximity value counts')
    print('----------------------------')
    print(housing["ocean_proximity"].value_counts())
    # <1H OCEAN     9136
    # INLAND        6551
    # NEAR OCEAN    2658
    # NEAR BAY      2290
    # ISLAND           5
    # Name: ocean_proximity, dtype: int64

def discover_visualize_data():
    print('--------------------------------------------------------------')
    print('discover经纬度组合，看看房子的地址分布')
    print('----------------------------')
    # 经纬度决定了房子街区的位置
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.6)
    plot.show()
    print('--------------------------------------------------------------')
    print('discover每个特征和房子的相关系数，选相关系数高的单独绘图看看')
    print('----------------------------')
    # look特征与房价的相关系数
    housing_corr_matrix = housing.corr()
    print(housing_corr_matrix['median_house_value'].sort_values(ascending=False))
    # median_house_value    1.000000
    # median_income         0.690647 高相关
    # total_rooms           0.133989 有关系
    # housing_median_age    0.103706 有关系
    # households            0.063714
    # total_bedrooms        0.047980
    # population           -0.026032
    # longitude            -0.046349
    # latitude             -0.142983 有负相关
    # 进一步探索相关系数比较高的几个特征两两之间的关系
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plot.show()
    # 房价和平均收入正相关无疑了
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plot.axis([0, 16, 0, 550000])
    plot.show()
    print('--------------------------------------------------------------')
    print('尝试一些组合特征是不是会更好')
    print('----------------------------')
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]
    housing["longitude_latitude"] = (housing["longitude"] + housing["latitude"])
    housing_corr_matrix = housing.corr()
    print(housing_corr_matrix['median_house_value'].sort_values(ascending=False))
    # median_house_value          1.000000
    # median_income               0.690647 高正相关
    # rooms_per_household         0.158485 崛起的新组合特征
    # total_rooms                 0.133989
    # housing_median_age          0.103706
    # households                  0.063714 可以去掉
    # total_bedrooms              0.047980 可以去掉
    # population_per_household   -0.022030 可以去掉
    # population                 -0.026032 可以去掉
    # longitude                  -0.046349 可以去掉
    # latitude                   -0.142983 可以去掉
    # bedrooms_per_room          -0.257419 崛起的新组合特征
    # longitude_latitude         -0.488857 崛起的新组合特征
    housing.plot(kind="scatter", x="longitude_latitude", y="median_house_value", alpha=0.1)
    plot.show()


# axis=1, reduce the columns, return a Series whose index is the original index 返回所有列不为空的index
# 与axis=1 对应的是 axis=0，reduce the index, return a Series whose index is the original column labels 返回所有数据不为空的列名
# 与any() 之对应的是 all()， any表示任意一个True则返回True，all表示任意一个为False则返回False
get_no_null_data = lambda df: df[df.isnull().any(axis=1)]


if __name__ == "__main__":
    pd.set_option('display.width', 1000)  # 设置字符显示宽度
    pd.set_option('display.max_columns', None) # 打印所有列，类似的max_rows打印所有行
    '''
        加载数据，并将数据划分成训练数据、测试数据
    '''
    housing = load_data()
    # quick_look_data()
    # 如果你想要安装收入就行层次取样就用StratifiedShuffleSplit
    # # Divide by 1.5 to limit the number of income categories
    # housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
    # # Label those above 5 as 5
    # housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=13)
    train_set_size = len(train_set)
    test_set_size = len(test_set)
    print('train set count', train_set_size, ', percent', train_set_size*1.0/(train_set_size+test_set_size),
          '\ntest set count',test_set_size, ', percent', test_set_size*1.0/(train_set_size+test_set_size), '\n')
    housing = train_set.copy()
    '''
        探索特征之间的相关性
    '''
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)  # drop labels for training set
    # discover_visualize_data()
    # housing["rooms_per_household"] = housing["total_rooms"] / housing["householdss"]
    # housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    # housing["population_per_household"] = housing["population"] / housing["households"]
    # housing["longitude_latitude"] = (housing["longitude"] + housing["latitude"])

    ## NULL值处理
    # print(get_no_null_data(housing).head()) # 查看有空值的特征数据
    imputer = SimpleImputer(strategy="median") # 空值用中位数替换
    housing_num = housing.select_dtypes(include=[np.number])
    imputer.fit(housing_num)
    # print(imputer.statistics_) # equal print(housing_num.median().values)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=list(housing.index.values))
    # print(housing_tr.head())
    # print(housing_labels.head())

    ## 字符串数值处理
    housing_cat = housing[['ocean_proximity']]
    print(housing_cat.head(10))
    housing_cat_1hot = LabelBinarizer().fit_transform(housing_cat)
    print(housing_cat_1hot)
