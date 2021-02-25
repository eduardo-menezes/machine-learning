## importing labraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Loading the data
data = pd.read_csv("Position_Salaries.csv")
x = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

# We must reshape "y" beacause the StandardScale functions expects
# a 2D array
y = y.reshape(len(y),1)

## There's no explicit relation between the predicted variable and the
#  predicters, so we must scale the features because now, there isn't
#  the coefficients to balance the values like we had in simple, multiple
#  or polynomial regression.

from sklearn.preprocessing import StandardScaler

## We have to create 2 objects from StandardScaler because it calculates
#  the mean and std for x variable and if we use the same object it will
#  use the mean and std of the x variable, in the y variable. That's incorrect
#  because both variables have different mean and std.
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)

## Training the model
#  we'll need a class that handles this kind of prediction, and it is SVM
from sklearn.svm import SVR
# Creating an object of this class and defining the kernel, i e, (replace a dot product for
# a non-linear function)
regressor = SVR(kernel="rbf")

# The training itself
# The .fit method expects the "y value" in 1D array, so we use np.ravel to do this
# the np.ravel, flattens the array
regressor.fit(x,np.ravel(y))
## Predicting new results
#  By the fact our data is scaled, the input predictions must be scaled too.
#  This scaling method must be the one which were used in the matrix of features
#  in our case here, (sc_x)
#  regressor.predict(sc_x.transform([[6.5]])) # here, the predicted salary is in the scale that
                                           # was applied to y
# therefore, instead of getting salary around 160K, we'll get a value around y range of values
# So we have to reverse the scale of y to obtain the correct salary value and this method is
# (inverse_transform). It's an object of "sc_y"
predicted_value = sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))
print(predicted_value)

## Visualizing the results of SVR
plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y),color="red")
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x)),color="blue")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

## The SVR doesn't catch the outliers, that's why the highest point didn't
#  got into the curve. It's preventing the overfitting.

## In SVR, we can choose linear kernels for data that presents linear relationships ("linear)
## and non-linear kernels for data that presents non-linear relationships. ("rbf","polynomial", etc)