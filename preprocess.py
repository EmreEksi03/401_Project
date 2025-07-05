from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.functions import dayofweek

# Spark oturumu başlat
spark = SparkSession.builder \
    .appName("NYC Cab Revenue Preprocessing") \
    .getOrCreate()

# Veri yolu (örnek CSV yolu)
data_path = "NYCTaxiDataset/data.csv"

# Veriyi oku
df = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path)

# Zaman tipini dönüştür
df = df.withColumn("pickup_datetime", to_timestamp("tpep_pickup_datetime")) \
       .withColumn("dropoff_datetime", to_timestamp("tpep_dropoff_datetime"))

# Saat, gün, ay, hafta sonu çıkar
df = df.withColumn("hour", hour("pickup_datetime")) \
       .withColumn("day_of_week", dayofweek("pickup_datetime")) \
       .withColumn("month", month("pickup_datetime")) \
       .withColumn("is_weekend", (col("day_of_week") >= 6).cast("int"))

# Gece yolculuğu? (örnek: 20:00 - 06:00)
df = df.withColumn("is_night", when((col("hour") >= 20) | (col("hour") < 6), 1).otherwise(0))

# Hız hesapla (miles per hour)
df = df.withColumn("trip_duration_min", (unix_timestamp("dropoff_datetime") - unix_timestamp("pickup_datetime")) / 60)
df = df.withColumn("speed_mph", col("trip_distance") / (col("trip_duration_min") / 60))

# Tatil bilgisi yer tutucu (external dataset ile eşleştirerek yapılmalı)
# Bu örnekte sadece yılbaşı kontrolü yapılıyor
df = df.withColumn("is_holiday", when(date_format("pickup_datetime", "MM-dd") == "01-01", 1).otherwise(0))

# Congestion surcharge varsa, yoğunluk var mı?
df = df.withColumn("congestion_happening", when(col("congestion_surcharge") > 0, 1).otherwise(0))

# Bahşiş oranı
df = df.withColumn("tip_ratio", 
                   when(col("fare_amount") > 0, col("tip_amount") / col("fare_amount")) \
                   .otherwise(0.0))

# Grup yolculuğu mu?
df = df.withColumn("is_group_ride", (col("passenger_count") >= 2).cast("int"))

# Uzak yolculuk mu?
df = df.withColumn("is_long_trip", (col("trip_distance") >= 10).cast("int"))

# PU-DO eşleşme kolonu
df = df.withColumn("location_pair", concat_ws("-", col("PULocationID").cast("string"), col("DOLocationID").cast("string")))

# Mil başına ücret
df = df.withColumn("fare_per_mile", 
                   when(col("trip_distance") > 0, col("fare_amount") / col("trip_distance")) \
                   .otherwise(0.0))

# İlk 10 satırı yazdır
# df.select("pickup_datetime", "hour", "is_night", "is_weekend", "speed_mph", "is_holiday", "congestion_happening").show(10)

output_path = "processed_output"  # bu klasör oluşacak
df.write.option("header", True).mode("overwrite").csv(output_path)

spark.stop()
