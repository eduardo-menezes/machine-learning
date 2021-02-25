## Loading the labraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Loading the data
data = pd.read_csv("Mall_Customers.csv")
x = data.iloc[:,[3,4]].values

## Using Dendograms to find the optimal number of clustes
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x,method="ward"))
plt.title("Dendogram")
plt.xlabel("Customer - Observation points")
plt.ylabel("Euclidean Distance")
plt.show()

#Training the model
from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5,affinity="euclidean", linkage="ward")
predicted_cluster = cluster.fit_predict(x)
print(predicted_cluster)

## Plotting the results
plt.scatter(x[predicted_cluster == 0,0],x[predicted_cluster == 0,1],s=100, c="red",label="Cluster 1")
plt.scatter(x[predicted_cluster == 1,0],x[predicted_cluster == 1,1],s=100, c="blue",label="Cluster 2")
plt.scatter(x[predicted_cluster == 2,0],x[predicted_cluster == 2,1],s=100, c="green",label="Cluster 3")
plt.scatter(x[predicted_cluster == 3,0],x[predicted_cluster == 3,1],s=100, c="cyan",label="Cluster 4")
plt.scatter(x[predicted_cluster == 4,0],x[predicted_cluster == 4,1],s=100, c="magenta",label="Cluster 5")
plt.title("Clusters of custumers")
plt.xlabel("Annual Income (K$) ")
plt.ylabel("Spending Score (1 - 100) ")
plt.legend()
plt.show()