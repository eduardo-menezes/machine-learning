## HOW to use and WHY use the decision tree regression ?
## We don't fave to scale the variables because the algorithm splits the data and it doesn't have
#  equations like the previous models.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Loading the data
data = pd.read_csv("Position_Salaries.csv")
x = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

## Training the model
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

## Predicting values
predicted_value = regressor.predict([[6.5]])
print(predicted_value)

## As we can see, the prediction is not quite good, that's because the Decision Tree Regression
#  doesn't work well with single feature dataset, it's more adapted to many features, to datasets
#  with higher dimensions
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()