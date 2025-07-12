from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

print(sc._jvm.org.apache.hadoop.util.VersionInfo.getVersion())