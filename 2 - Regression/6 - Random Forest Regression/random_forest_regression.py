## importing labraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Loading the data
data = pd.read_csv("Position_Salaries.csv")
x = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

## Trainning the model
from sklearn.ensemble import RandomForestRegressor

# Creating an object from this class
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x,y)

## Predicting new values
predicted_value = regressor.predict([[6.5]])
print(predicted_value)

## Visualising the results
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()