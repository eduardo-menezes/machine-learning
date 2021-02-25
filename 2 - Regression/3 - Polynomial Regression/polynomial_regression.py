## importing labraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Loading the data
data = pd.read_csv("Position_Salaries.csv")
x = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

## Building a linear model to compare with the
#  polynomial
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

## Now let's build the polynomial regressor
#  We need to import the library to preprocess the features into
#  "features-to-the-power-of n, i. e. x,x²,x³... x^n "
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)

## Ok, we've done the "features-to-the-power-of n, i. e. x,x²,x³... x^n "
#  now we must make a linear combination with the features and the coeficients
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(x_poly,y)

'''
## Now let's visualize the Linear Regression results
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg.predict(x),color="blue")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
'''

'''
## It's very clear that the model didn't fit so well to this data
#  Let's plot the polynomial regression
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg_poly.predict(x_poly),color="blue")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
'''

## It's easy to see that now, the polynomial regressor is better fitted than
#  the linear one. Now let's predict for a region manager in the point of 6.5
#  if he's bleffing or not
pred = np.array(6.5)
ylin_pred = lin_reg.predict([[6.5]])
x_poly = poly_reg.fit_transform([[6.5]])
ypoly_pred = lin_reg_poly.predict(x_poly)
print(ylin_pred,ypoly_pred)