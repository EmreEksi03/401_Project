import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim
from pyspark.sql.types import DoubleType

# Start Spark session
spark = SparkSession.builder \
    .appName("OcularDiseaseRecognition") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("Spark session created.")
print("Spark Version:", spark.version)

# Load dataset
df = spark.read.csv("NYCTaxiDataset/data.csv", header=True, inferSchema=True)

# Clean column names: strip whitespace
df = df.toDF(*[col_name.strip() for col_name in df.columns])

# Optional: trim string values
for col_name, dtype in df.dtypes:
    if dtype == "string":
        df = df.withColumn(col_name, trim(col(col_name)))

# Drop rows with nulls in essential columns
# required_columns = ["fare_amount", "tip_amount"]
#df_clean = df.na.drop(subset=required_columns)

# Drop rows with nulls
df_clean = df.na.drop()

# Optional: remove duplicates
df_clean = df_clean.dropDuplicates()

# Save as Parquet
df_clean.write.mode("overwrite").parquet("clean_NYC_Taxi.parquet")
df.printSchema()

print("Cleaning complete. Data saved to 'clean_NYC_Taxi.parquet'")
