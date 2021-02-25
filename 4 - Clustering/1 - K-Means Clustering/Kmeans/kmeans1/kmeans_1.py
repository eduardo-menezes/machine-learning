# importing the libraries that'll be used
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# defining all the plots to seaborn
sns.set()

# loading the data
data = pd.read_csv("3.01. Country clusters.csv")
print(data)

# Plotting the data
plt.scatter(data["Longitude"],data["Latitude"])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

# Selecting the features
x = data.iloc[:,1:3]
print(x)

# The clustering itself
kmeans = KMeans(3)
kmeans.fit(x)

# The predicted groups
identified_clusters = kmeans.fit_predict(x)
print(identified_clusters)

# Creating a dataframe to organize the resulting cluster
data_with_cluster = data.copy()
data_with_cluster["Cluster"] = identified_clusters
print(data_with_cluster)

# Plotting the results
plt.scatter(data_with_cluster["Longitude"],data_with_cluster["Latitude"],c=data_with_cluster["Cluster"],cmap="rainbow")
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()
