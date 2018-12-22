# !usr/bin/env python
# coding:utf-8

"""
房价预测
author: prucehuang
 email: 1756983926@qq.com
  date: 2018/12/22
"""
import numpy as np
import os
import sys
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.stats import expon, reciprocal

# file path
PROJECT_ROOT_DIR = sys.path[0] + '/../'
HOUSING_PATH = os.path.join(PROJECT_ROOT_DIR, 'datasets', 'housing')

def load_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    housing = pd.read_csv(csv_path)
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=13)
    train_set_size = len(train_set)
    test_set_size = len(test_set)
    print('train set count', train_set_size, ', percent', train_set_size * 1.0 / (train_set_size + test_set_size),
          '\ntest set count', test_set_size, ', percent', test_set_size * 1.0 / (train_set_size + test_set_size), '\n')
    train_set_labels = train_set["median_house_value"].copy()
    train_set = train_set.drop("median_house_value", axis=1)  # drop labels for training set
    test_set_labels = test_set["median_house_value"].copy()
    test_set = test_set.drop("median_house_value", axis=1)
    return train_set, train_set_labels, test_set, test_set_labels

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        # column index
        longitude_ix, latitude_ix, rooms_ix, bedrooms_ix, population_ix, household_ix = 0, 1, 3, 4, 5, 6

        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        longitude_latitude = X[:, longitude_ix] + X[:, latitude_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, longitude_latitude,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household, longitude_latitude]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

def feature_clear(housing):
    housing_num = housing.select_dtypes(include=[np.number])
    num_attribs = list(housing_num)
    cat_attribs = ['ocean_proximity']

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('one_hot', OneHotEncoder()),
    ])
    feature_pipeline = ColumnTransformer([
        ("num_pipeline", num_pipeline, num_attribs),
        ("cat_pipeline", cat_pipeline, cat_attribs),
    ])
    return feature_pipeline.fit(housing)

def display_score(model, housing_prepared, housing_labels):
    housing_predictions = model.predict(housing_prepared)
    # 平方误差
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print('mean_squared_error', tree_rmse)
    # 绝对值误差
    tree_mae = mean_absolute_error(housing_labels, housing_predictions)
    print('mean_absolute_error', tree_mae)
    # 交叉验证误差
    scores = cross_val_score(model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
    scores = np.sqrt(-scores)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print(pd.Series(scores).describe())

if __name__ == "__main__":
    pd.set_option('display.width', 1000)  # 设置字符显示宽度
    pd.set_option('display.max_columns', None)  # 打印所有列，类似的max_rows打印所有行

    '''
        加载数据，并将数据划分成训练数据、测试数据
    '''
    housing, housing_labels, test_set, test_set_labels = load_data()

    '''
        特征处理
    '''
    feature_pipeline = feature_clear(housing)
    housing_prepared = feature_pipeline.transform(housing)

    '''
        自动选择超参数          
    '''
    get_hyperparameters_way = 'RandomizedSearchCV'
    forest_reg = RandomForestRegressor(random_state=42)
    svm_reg = SVR()
    if get_hyperparameters_way == 'GridSearchCV':
        param_grid = [
            {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
            {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
             'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
        ]
        hyperparameters_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error',
                                              verbose=2, n_jobs=4)
    else:
        param_distribs = {
            'kernel': ['linear', 'rbf'],
            'C': reciprocal(20000, 200000),
            'gamma': expon(scale=1.0),
        }
        hyperparameters_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                verbose=2, n_jobs=4, random_state=42)
    hyperparameters_search.fit(housing_prepared, housing_labels)
    print(hyperparameters_search.best_params_)
    print(hyperparameters_search.best_estimator_)
    cvres = hyperparameters_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    print(pd.DataFrame(hyperparameters_search.cv_results_))

    '''
        从model分析特征的相关性
    '''
    feature_importances = hyperparameters_search.best_estimator_.feature_importances_
    extra_attribs = ['rooms_per_household', 'population_per_household', 'longitude_latitude', 'bedrooms_per_room']
    num_attribs = list(housing.select_dtypes(include=[np.number]))
    cat_one_hot_attribs = list(feature_pipeline.named_transformers_["cat_pipeline"]
                               .named_steps['one_hot'].categories_[0])
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    print(sorted(zip(feature_importances, attributes), reverse=True))
    print(indices_of_top_k(feature_importances, 5))

    # '''
    #     测试集预测
    # '''
    # final_model = hyperparameters_search.best_estimator_
    #
    # y_test = test_set["median_house_value"].copy()
    # X_test = test_set.drop("median_house_value", axis=1)
    # X_test_prepared = feature_pipeline.transform(X_test)
    #
    # print('------------------final model -----------------------')
    # display_score(final_model, X_test_prepared, y_test)

