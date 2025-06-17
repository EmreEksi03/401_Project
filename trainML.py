import shutil
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, unix_timestamp
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Start Spark session with more memory and tuned configs
spark = SparkSession.builder \
    .appName("NYC Taxi Fare & Time Model") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.memory", "8g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.5") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

# Set log level to ERROR to reduce verbosity
spark.sparkContext.setLogLevel("ERROR")

print("[OK] Spark master:", spark.sparkContext.master)
print("[OK] Executor count:", spark._jsc.sc().getExecutorMemoryStatus().size())

# Load cleaned data
df = spark.read.parquet("clean_NYC_Taxi.parquet")

# Optional: Sample data to reduce size if needed (adjust fraction)
df = df.sample(fraction=0.3, seed=42)  # 30% sample

# Repartition to balance memory load
df = df.repartition(100)

# ========== 1. Trip Duration in Minutes ==========
df = df.withColumn(
    "trip_duration_minutes",
    (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60
)

# ========== 2. Time Features ==========
df = df.withColumn("hour_of_day", hour("tpep_pickup_datetime"))
df = df.withColumn("day_of_week", dayofweek("tpep_pickup_datetime"))

# ========== 3. Assemble Features ==========
feature_cols = ["trip_distance", "hour_of_day", "day_of_week", "passenger_count"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_model = assembler.transform(df)

# ========== 4. Split into Train/Test ==========
train_df, test_df = df_model.randomSplit([0.8, 0.2], seed=42)
test_df.write.mode("overwrite").parquet("models/test_data.parquet")
print(f"[OK] Training dataset count: {train_df.count()}, Test dataset count: {test_df.count()}")

# ========== 5. Train Models ==========

# Fare Model
fare_model = LinearRegression(featuresCol="features", labelCol="fare_amount")
fare_model_fit = fare_model.fit(train_df)

# Trip Duration Model
time_model = LinearRegression(featuresCol="features", labelCol="trip_duration_minutes")
time_model_fit = time_model.fit(train_df)

# ========== 6. Evaluate Models ==========

evaluator = RegressionEvaluator(metricName="rmse")

fare_predictions = fare_model_fit.transform(test_df)
fare_rmse = evaluator.evaluate(fare_predictions, {evaluator.labelCol: "fare_amount", evaluator.predictionCol: "prediction"})
print(f"[OK] Fare Model RMSE on test data: {fare_rmse:.4f}")

time_predictions = time_model_fit.transform(test_df)
time_rmse = evaluator.evaluate(time_predictions, {evaluator.labelCol: "trip_duration_minutes", evaluator.predictionCol: "prediction"})
print(f"[OK] Trip Duration Model RMSE on test data: {time_rmse:.4f}")

# ========== 7. Overwrite-Safe Save ==========

def overwrite_save(model, path):
    if os.path.exists(path):
        shutil.rmtree(path)
    model.save(path)

overwrite_save(fare_model_fit, "models/fare_model")
overwrite_save(time_model_fit, "models/time_model")

print("[OK] Models trained, evaluated, and saved with overwrite.")
