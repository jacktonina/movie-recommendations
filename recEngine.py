# Import necessary packages
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
from config import Config

# Might have to uncomment?
# import os
# memory = '10g'
# pyspark_submit_args = ' --driver-memory ' + memory + ' pyspark-shell'
# os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args


# Create Spark object
def spark(SparkSession):
    spark = SparkSession\
        .builder\
        .appName("example-spark")\
        .config("spark.sql.crossJoin.enabled", "true")\
        .getOrCreate()
    sc = SparkContext.getOrCreate()
    pyspark = SQLContext(sc)
    return pyspark


def read_data(mov, rat, sqlContext):
    movies = sqlContext.read.csv(mov, header=True)

    ratings = sqlContext.read.csv(rat, header=True)
    ratings = ratings.drop('timestamp')
    ratings = ratings.selectExpr("cast(userId as int) userId",
        "cast(movieId as int) movieId",
        "cast(rating as int) rating")
    return movies, ratings

def split(movies, ratings):
    # Split ratings and movies dataframes into training and testing sets
    movies_train, movies_test = movies.randomSplit([0.8, 0.2])
    ratings_train, ratings_test = ratings.randomSplit([0.8, 0.2])
    return movies_train, movies_test, ratings_train, ratings_test


def model(ratings_train):
    als = ALS(maxIter=5, regParam=0.09, rank=25, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop", nonnegative=True)
    model = als.fit(ratings_train)
    return model


def predict(model, ratings_test):
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    predictions = model.transform(ratings_test)

    rmse = evaluator.evaluate(predictions)
    print("RMSE= "+str(rmse))
    return predictions

#predictions.show()


def recommendations(model, ratings):
    user_recs = model.recommendForAllUsers(20).show(5)
    user_recs.show()

    # NEW
    recs = model.recommendForAllUsers(10).toPandas()
    nrecs = recs.recommendations.apply(pd.Series) \
        .merge(recs, right_index=True, left_index=True) \
        .drop(["recommendations"], axis=1) \
        .melt(id_vars=['userId'], value_name="recommendation") \
        .drop("variable", axis=1) \
        .dropna()
    nrecs = nrecs.sort_values('userId')
    nrecs = pd.concat([nrecs['recommendation'].apply(pd.Series), nrecs['userId']], axis=1)
    nrecs.columns = [

        'movieId',
        'rating',
        'userId'

    ]
    md = ratings.select(ratings['userId'], ratings['userId'], ratings['movieId'], ratings['movieId'])
    md = md.toPandas()
    dict1 = dict(zip(md['userId'], md['userId']))
    dict2 = dict(zip(md['movieId'], md['movieId']))
    nrecs['userId'] = nrecs['userId'].map(dict1)
    nrecs['movieId'] = nrecs['movieId'].map(dict2)
    nrecs = nrecs.sort_values('userId')
    nrecs.reset_index(drop=True, inplace=True)
    new = nrecs[['userId', 'movieId', 'rating']]
    new['recommendations'] = list(zip(new.movieId, new.rating))
    res = new[['userId', 'recommendations']]
    res_new = res['recommendations'].groupby([res.userId]).apply(list).reset_index()
    print(res_new)
    return res_new

def main():
    sqlContext = spark(SparkSession)
    read_data("data/movies.csv", "data/ratings.csv", sqlContext=sqlContext)
    split(movies=-movies, ratings=ratings)
    predict(model=model, ratings_test=ratings_test)
    recommendations(model=model, training=ratings)


if __name__ == "__main__":
    main()

# # NEW
# recs = model.recommendForAllUsers(10).toPandas()
# nrecs = recs.recommendations.apply(pd.Series) \
#     .merge(recs, right_index=True, left_index=True) \
#     .drop(["recommendations"], axis=1) \
#     .melt(id_vars=['reviewerID_index'], value_name="recommendation") \
#     .drop("variable", axis=1) \
#     .dropna()
# nrecs = nrecs.sort_values('reviewerID_index')
# nrecs = pd.concat([nrecs['recommendation'].apply(pd.Series), nrecs['reviewerID_index']], axis=1)
# nrecs.columns = [
#
#     'ProductID_index',
#     'Rating',
#     'UserID_index'
#
# ]
# md = transformed.select(transformed['reviewerID'], transformed['reviewerID_index'], transformed['asin'],
#                         transformed['asin_index'])
# md = md.toPandas()
# dict1 = dict(zip(md['reviewerID_index'], md['reviewerID']))
# dict2 = dict(zip(md['asin_index'], md['asin']))
# nrecs['reviewerID'] = nrecs['UserID_index'].map(dict1)
# nrecs['asin'] = nrecs['ProductID_index'].map(dict2)
# nrecs = nrecs.sort_values('reviewerID')
# nrecs.reset_index(drop=True, inplace=True)
# new = nrecs[['reviewerID', 'asin', 'Rating']]
# new['recommendations'] = list(zip(new.asin, new.Rating))
# res = new[['reviewerID', 'recommendations']]
# res_new = res['recommendations'].groupby([res.reviewerID]).apply(list).reset_index()
# print(res_new)

# ----------


# # Creates ALS model
# rank = 10
# iterations = 10
# model = ALS.train(ratings_train, rank, iterations)

#recs = model.recommendProducts(4169, 5)
# predictions = model.transform(ratings_test.select(["user", "item"]))
# print(predictions)
# print(type(predictions))

#titles = movies.rdd.map(lambda p: (p[0], p[1]))

# print('Titles')
# top_10 = titles.take(10)
# print(top_10)


# # dropping the ratings on the tests data
# test_data = ratings_test.rdd.map(lambda p: (p[0], p[1]))
# predictions = model.predictAll(test_data).map(lambda r: ((r[0], r[1]), r[2]))
#
# # joining the prediction with the original test dataset
# rates_vs_pred = ratings_test.rdd.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
#
# # calculating error
# MSE = rates_vs_pred.map(lambda r: (r[1][0] - r[1][1])**2).mean()
# print("Mean Squared Error = " + str(MSE))


# # With dataframes
# from pyspark.ml.recommendation import ALS
#
# from pyspark.sql.types import FloatType
# from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.sql.functions import col
#
#

#
# ratings = pd.read_csv("data/ratings.csv")
#
# ratings_train = ratings.sample(frac=0.75, random_state=1)
# ratings_test = ratings.drop(ratings_train.index)
#
# print(len(ratings_train))
# print(len(ratings_test))
#
# als = mlALS(rank=5, maxIter=10, seed=0)
# model = als.fit(ratings_train.select(["user", "item", "rating"]))


