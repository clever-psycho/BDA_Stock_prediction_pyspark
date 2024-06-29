from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, avg
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt

# Create a Spark session
spark = SparkSession.builder.appName("StockPriceAnalysis").getOrCreate()

# Load the CSV file into a DataFrame
df = spark.read.csv("AAPL.csv", header=True, inferSchema=True)

# Convert the date column to a date type and drop rows with missing values
df = df.withColumn("date", to_date(col("Date"), "yyyy-MM-dd"))
df = df.dropna()
df.show(5)

# Create a window specification
windowSpec = Window.partitionBy().orderBy("date").rowsBetween(-9, 0)

# Add a moving average column
df = df.withColumn("moving_avg", avg("Close").over(windowSpec))
df.show(5)

# Prepare the data for training, include date column for later use
assembler = VectorAssembler(inputCols=["moving_avg"], outputCol="features")
data = assembler.transform(df)
data = data.select("date", "features", "Close")

# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2])

# Create and train the Linear Regression model
lr = LinearRegression(labelCol="Close")
model = lr.fit(train_data)

# Make predictions
predictions = model.transform(test_data)
predictions.show(5)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="Close", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Convert predictions to a Pandas DataFrame for visualization
pandas_df = predictions.select("date", "Close", "prediction").toPandas()

# Plot the actual vs predicted stock prices
plt.figure(figsize=(14, 7))
plt.plot(pandas_df["date"], pandas_df["Close"], label="Actual")
plt.plot(pandas_df["date"], pandas_df["prediction"], label="Predicted")
plt.legend()
plt.show()