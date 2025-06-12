from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, radians, sin, cos, atan2, sqrt,
    hour, dayofweek, to_timestamp, unix_timestamp
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Start Spark session
spark = SparkSession.builder \
    .appName("NYC Taxi Fare & Time Model") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Load cleaned data from Parquet
df = spark.read.parquet("clean_NYC_Taxi.parquet")

# ========== 1. Haversine Distance ==========
#def add_haversine_distance(df):
#    R = 6371  # Earth radius in km

#    df = df.withColumn("pickup_lat_rad", radians(col("pickup_latitude"))) \
#           .withColumn("pickup_lon_rad", radians(col("pickup_longitude"))) \
#           .withColumn("dropoff_lat_rad", radians(col("dropoff_latitude"))) \
#           .withColumn("dropoff_lon_rad", radians(col("dropoff_longitude")))

#    df = df.withColumn("delta_lat", col("dropoff_lat_rad") - col("pickup_lat_rad")) \
#           .withColumn("delta_lon", col("dropoff_lon_rad") - col("pickup_lon_rad"))

#    df = df.withColumn("a", sin(col("delta_lat") / 2) ** 2 +
#                            cos(col("pickup_lat_rad")) * cos(col("dropoff_lat_rad")) *
#                            sin(col("delta_lon") / 2) ** 2)

#    df = df.withColumn("c", 2 * atan2(sqrt(col("a")), sqrt(1 - col("a"))))
#    df = df.withColumn("haversine_distance_km", R * col("c"))

#    return df.drop("pickup_lat_rad", "pickup_lon_rad", "dropoff_lat_rad", 
#                   "dropoff_lon_rad", "delta_lat", "delta_lon", "a", "c")

# df = add_haversine_distance(df)

# ========== 2. Trip Duration in Minutes ==========
df = df.withColumn(
    "trip_duration_minutes",
    (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60
)

# ========== 3. Add Time Features ==========
df = df.withColumn("hour_of_day", hour("tpep_pickup_datetime"))
df = df.withColumn("day_of_week", dayofweek("tpep_pickup_datetime"))

# ========== 4. Assemble Feature Vector ==========
feature_cols = ["trip_distance", "hour_of_day", "day_of_week", "passenger_count"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_model = assembler.transform(df)


# ========== 5. Train Models ==========

# --- Fare Model ---
fare_model = LinearRegression(featuresCol="features", labelCol="fare_amount")
fare_model_fit = fare_model.fit(df_model)

# --- Trip Time Model ---
time_model = LinearRegression(featuresCol="features", labelCol="trip_duration_minutes")
time_model_fit = time_model.fit(df_model)

# ========== 6. Save Models ==========
fare_model_fit.save("models/fare_model")
time_model_fit.save("models/time_model")

print("✅ Models trained and saved.")
