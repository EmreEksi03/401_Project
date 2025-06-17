from pyspark.sql import SparkSession

# Start Spark session
spark = SparkSession.builder \
    .appName("PrintSchema") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Load cleaned Parquet data
df = spark.read.parquet("clean_NYC_Taxi.parquet")

# Print schema
df.printSchema()

print("Schema printed successfully.")
