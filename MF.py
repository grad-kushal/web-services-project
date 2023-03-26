import pyspark as py
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql import SparkSession


def load_data():
    spark = SparkSession.builder.appName("ALS_example").getOrCreate()


    data = spark.read.csv("example.csv", header=True, inferSchema=True)


    (training, test) = data.randomSplit([0.8, 0.2], seed=42)


    als = ALS(maxIter=10, regParam=0.01, userCol="mashup_id", itemCol="api_id", ratingCol="probability",
              coldStartStrategy="drop")

    # Fit the model to the training data
    model = als.fit(training)

    # Evaluate the model on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="probability", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"Root-mean-square error = {rmse}")


    userRecs = model.recommendForAllUsers(10)


    userRecs.show(5, truncate=False)








if __name__ == "__main__":
    load_data()