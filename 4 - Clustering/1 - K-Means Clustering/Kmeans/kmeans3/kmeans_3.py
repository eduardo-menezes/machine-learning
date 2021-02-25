import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
sns.set()

# Loading data
data = pd.read_csv("3.12. Example.csv")
print(data)

# Plotting the data
plt.scatter(data["Satisfaction"], data["Loyalty"])
plt.xlabel("Satisfaction")
plt.ylabel("Loyalty")
#plt.show()

# Selecting the feature
x = data.copy()

#The clustering
kmeans = KMeans(2)
kmeans.fit(x)

#Clustering results
clusters = x.copy()
clusters["cluster_pred"] = kmeans.fit_predict(x)

#Plotting the results
plt.scatter(clusters["Satisfaction"], clusters["Loyalty"], c=clusters["cluster_pred"],cmap="rainbow")
plt.ylabel("Loyalty")
plt.xlabel("Satisfaction")
#plt.show()

'''As observed, there's a problem when we cluster without normalizing the data
because only satisfaction is considered '''

#Normalizing the data
from sklearn import preprocessing
x_scaled = preprocessing.scale(x)

''' as we don't know the numbers os groups we need, we'll use the elbow method '''
wcss = []

for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,10), wcss)
plt.xlabel("Number of clusters")
plt.ylabel("wcss")
plt.show()

# Now, we should explore cluster solution and select the best number os clusters
kmeans_new = KMeans(4)
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new["cluster_pred"] = kmeans_new.fit_predict(x_scaled)
print(clusters_new)

#Plotting the new data
plt.scatter(clusters_new["Satisfaction"],clusters_new["Loyalty"],c=clusters_new["cluster_pred"],cmap="rainbow")
plt.ylabel("Loyalty")
plt.xlabel("Satisfaction")
plt.show()
