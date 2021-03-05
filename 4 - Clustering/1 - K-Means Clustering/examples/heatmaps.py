# importing the libraries
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Loading the data

data = pd.read_csv("Country clusters standardized.csv", index_col="Country")
xscaled = data.copy()
xscaled = xscaled.drop(["Language"], axis=1)
print(xscaled)

# Plotting the heatmap
sns.clustermap(xscaled)
plt.show()

