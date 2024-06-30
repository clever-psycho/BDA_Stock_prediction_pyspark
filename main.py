from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, avg
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta

# Create a Spark session
spark = SparkSession.builder.appName("StockPriceAnalysis").getOrCreate()

# Load the CSV file into a DataFrame
df = spark.read.csv("NFLX.csv", header=True, inferSchema=True)

# Convert the date column to a date type and drop rows with missing values
df = df.withColumn("date", to_date(col("Date"), "yyyy-MM-dd"))
df = df.dropna()

# Create a window specification for moving averages
windowSpec = Window.partitionBy().orderBy("date").rowsBetween(-9, 0)

# Add moving average column
df = df.withColumn("moving_avg", avg("Close").over(windowSpec))

# Calculate 50-day and 200-day moving averages
windowSpec50 = Window.partitionBy().orderBy("date").rowsBetween(-49, 0)
windowSpec200 = Window.partitionBy().orderBy("date").rowsBetween(-199, 0)
df = df.withColumn("moving_avg_50", avg("Close").over(windowSpec50))
df = df.withColumn("moving_avg_200", avg("Close").over(windowSpec200))

# Prepare the data for training
assembler = VectorAssembler(inputCols=["moving_avg"], outputCol="features")
data = assembler.transform(df)
data = data.select("date", "features", "Close")

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2])

# Create and train the Linear Regression model
lr = LinearRegression(labelCol="Close")
model = lr.fit(train_data)

# Make predictions on the test set
predictions = model.transform(test_data)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Convert predictions to a Pandas DataFrame for visualization
pandas_df = predictions.select("date", "Close", "prediction").toPandas()

# Add the next day's date for prediction
last_date = df.select("date").toPandas()["date"].max()
next_date = last_date + timedelta(days=1)
future_df = pd.DataFrame([next_date], columns=["date"])
future_df["moving_avg"] = pandas_df["prediction"].iloc[-10:].mean()

# Create Spark DataFrame from the next day's date
future_spark_df = spark.createDataFrame(future_df)

# Prepare future data for prediction
future_data = assembler.transform(future_spark_df)
future_data = future_data.select("date", "features")

# Predict future price
future_predictions = model.transform(future_data)
future_pandas_df = future_predictions.select("date", "prediction").toPandas()

# Plot the actual vs predicted stock prices and moving averages
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(pandas_df["date"], pandas_df["Close"], label="Actual")
ax.plot(pandas_df["date"], pandas_df["prediction"], label="Predicted")
ax.plot(df.select("date").toPandas()["date"], df.select("moving_avg_50").toPandas()["moving_avg_50"], label="50-Day MA", linestyle='--')
ax.plot(df.select("date").toPandas()["date"], df.select("moving_avg_200").toPandas()["moving_avg_200"], label="200-Day MA", linestyle='--')
plt.legend()

# Highlight the future predicted value
row = future_pandas_df.iloc[0]
ax.annotate(f'Predicted: {row["prediction"]:.2f} on {row["date"].strftime("%Y-%m-%d")}',
            xy=(row["date"], row["prediction"]),
            xytext=(row["date"] - timedelta(days=30), row["prediction"] + 10),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.8))

plt.grid(True)
plt.show()
