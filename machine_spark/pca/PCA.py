# !/usr/bin/python
# coding:utf-8

from __future__ import print_function

# $example on$
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
# $example off$
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("PCAExample")\
        .getOrCreate()

    # $example on$
    # 稀疏矩阵
    # sparse(size, *args): 更方便的创建系数矩阵, args是 a list of (index, value) pairs
    # Vectors.sparse(5, [(1, 1.0), (3, 3.0)] 实际上是 (0.0, 1.0, 0..0, 3.0, 0.0)
    # 密集矩阵
    # dense(*elements): Create a dense vector of 64-bit floats from a Python list or numbers.
    data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
            (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
            (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
    df = spark.createDataFrame(data, ["features"])

    pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
    model = pca.fit(df)
    result = model.transform(df)
    result.show(truncate=False)
    # $example off$

    spark.stop()
