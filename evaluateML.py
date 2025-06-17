from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator

# Spark session
spark = SparkSession.builder \
    .appName("Evaluate NYC Taxi Models") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()
# Set log level to ERROR to reduce verbosity
spark.sparkContext.setLogLevel("ERROR")

# Load test data
test_df = spark.read.parquet("models/test_data.parquet")
print(f"[OK] Loaded test data with {test_df.count()} rows")

# Load saved models
fare_model = LinearRegressionModel.load("models/fare_model")
time_model = LinearRegressionModel.load("models/time_model")

# Predict
fare_predictions = fare_model.transform(test_df)
time_predictions = time_model.transform(test_df)

# Evaluate
evaluator = RegressionEvaluator()

# Fare model metrics
evaluator.setLabelCol("fare_amount")
evaluator.setPredictionCol("prediction")
fare_rmse = evaluator.evaluate(fare_predictions, {evaluator.metricName: "rmse"})
fare_r2 = evaluator.evaluate(fare_predictions, {evaluator.metricName: "r2"})

# Trip duration model metrics
evaluator.setLabelCol("trip_duration_minutes")
time_rmse = evaluator.evaluate(time_predictions, {evaluator.metricName: "rmse"})
time_r2 = evaluator.evaluate(time_predictions, {evaluator.metricName: "r2"})

print(f"Fare Model - RMSE: {fare_rmse:.4f}, R²: {fare_r2:.4f}")
print(f"Trip Duration Model - RMSE: {time_rmse:.4f}, R²: {time_r2:.4f}")
