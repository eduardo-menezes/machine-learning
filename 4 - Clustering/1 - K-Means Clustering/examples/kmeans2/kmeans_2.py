# libraries

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np

#Again, loading data
data = pd.read_csv("3.01. Country clusters.csv")

# Mapping the data
data_mapped = data.copy()
data_mapped["Language"] = data_mapped["Language"].map({"English":0, "French":1, "German":2})

# Preparing the features
x = data_mapped.iloc[:,3:4]
print(x)

#The clustering itself
kmeans = KMeans(3)
kmeans.fit(x)

# Identifying the clustering results
identified_cluster = kmeans.fit_predict(x)
print(identified_cluster)

# Building a dataframe to resume the results
data_with_cluster = data_mapped.copy()
data_with_cluster["Cluster"] = identified_cluster
print(data_with_cluster)

#Plotting the results
plt.scatter(data_with_cluster["Longitude"],data_with_cluster["Latitude"], c=data_with_cluster["Cluster"], cmap="rainbow")
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()


