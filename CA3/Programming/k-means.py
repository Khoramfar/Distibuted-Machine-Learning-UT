from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import mean, min, max, variance
import matplotlib.pyplot as plt
import time


if __name__ == "__main__":
    start_time = time.time()
    spark = SparkSession\
        .builder\
        .appName("KMeans-Khoramfar")\
        .getOrCreate()
    
    file_path = "hdfs://raspberrypi-dml0:9000/810102129/customers.csv"  
    data = spark.read.csv(file_path, header=True, inferSchema=True)


    stats = data.select(
        mean("Annual Income (k$)").alias("Mean"),
        min("Annual Income (k$)").alias("Min"),
        max("Annual Income (k$)").alias("Max"),
        variance("Annual Income (k$)").alias("Variance")
    )
    print("Statistics:")
    stats.show()

    assembler = VectorAssembler(
        inputCols=["Annual Income (k$)", "Spending Score (1-100)"], outputCol="features"
    )
    dataset = assembler.transform(data)

    kmeans = KMeans().setK(5).setSeed(1)  
    model = kmeans.fit(dataset)

    predictions = model.transform(dataset)

    pandas_df = predictions.select("Annual Income (k$)", "Spending Score (1-100)", "prediction").toPandas()

    plt.figure(figsize=(10, 6))
    for cluster in pandas_df['prediction'].unique():
        cluster_data = pandas_df[pandas_df['prediction'] == cluster]
        plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f"Cluster {cluster}")

    plt.title("Clusters")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.legend()
    plt.show()
    plt.savefig("plot.png")

    spark.stop()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time: {elapsed_time:.2f} seconds")