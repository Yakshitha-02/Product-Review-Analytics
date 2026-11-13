
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, concat_ws, trim, substring, length
)

spark = SparkSession.builder \
    .appName("Prepare ML Training Sample") \
    .master("local[2]") \
    .config("spark.driver.memory", "1g") \
    .getOrCreate()

print("Reading reviews from HDFS...")

df = spark.read.parquet(
    "hdfs://localhost:9000/product_review_analytics/input/*.parquet"
)

reviews = df.withColumn(
    "sentiment",
    when(col("rating") <= 2, "Negative")
    .when(col("rating") == 3, "Neutral")
    .otherwise("Positive")
).withColumn(
    "review",
    trim(concat_ws(" ", col("title"), col("text")))
).filter(
    col("review").isNotNull() & (length(col("review")) > 0)
).select(
    substring(col("review"), 1, 500).alias("review"),
    col("sentiment")
)

print("Sampling 10,000 reviews from each class...")

negative = reviews.filter(
    col("sentiment") == "Negative"
).sample(False, 0.02, seed=42).limit(10000)

neutral = reviews.filter(
    col("sentiment") == "Neutral"
).sample(False, 0.05, seed=42).limit(10000)

positive = reviews.filter(
    col("sentiment") == "Positive"
).sample(False, 0.003, seed=42).limit(10000)

balanced = negative.union(neutral).union(positive)

output_path = (
    "file:///home/yakshu/ProductReviewAnalytics/dataset/ml_sample_json"
)

balanced.coalesce(1).write \
    .mode("overwrite") \
    .json(output_path)
print("ML sample saved successfully!")

spark.stop()
