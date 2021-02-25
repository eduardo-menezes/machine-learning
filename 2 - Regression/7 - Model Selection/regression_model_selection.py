import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


## Loading the dataset
data = pd.read_csv("Data.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
## Now, we must split our data into train and test data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

def MLR(xtrain, xtest, ytrain, ytest ):
    ''' Multiple Lienar Regression'''

    ## Let's train the model in the tarining test
    regressor = LinearRegression()
    regressor.fit(xtrain,ytrain)
    ypred = regressor.predict(xtest)
    score = r2_score(ytest, ypred)

    return [score, "Multiple Linear Regression"]


def PlmR(xtrain, xtest, ytrain, ytest ):

    ## Now let's build the polynomial regressor
    #  We need to import the library to preprocess the features into
    #  "features-to-the-power-of n, i. e. x,x²,x³... x^n "
    poly_reg = PolynomialFeatures(degree=4)
    xtrain_poly = poly_reg.fit_transform(xtrain)
    xtest_poly = poly_reg.transform(xtest)

    ## Ok, we've done the "features-to-the-power-of n, i. e. x,x²,x³... x^n "
    #  now we must make a linear combination with the features and the coeficients
    lin_reg_poly = LinearRegression()
    lin_reg_poly.fit(xtrain_poly, ytrain)
    ypred = lin_reg_poly.predict(xtest_poly)
    score = r2_score(ytest,ypred)
    #print("R2 for Polynomial Regression: ", score)
    return [score, "Polynomial Regression"]

def SVR(xtrain, xtest, ytrain, ytest ):

    # We must reshape "y" beacause the StandardScale functions expects
    # a 2D array
    ytrain = ytrain.reshape(len(ytrain), 1)

    ## There's no explicit relation between the predicted variable and the
    #  predicters, so we must scale the features because now, there isn't
    #  the coefficients to balance the values like we had in simple, multiple
    #  or polynomial regression.

    ## We have to create 2 objects from StandardScaler because it calculates
    #  the mean and std for x variable and if we use the same object it will
    #  use the mean and std of the x variable, in the y variable. That's incorrect
    #  because both variables have different mean and std.
    sc_x = StandardScaler()
    xscaled = sc_x.fit_transform(xtrain)

    sc_y = StandardScaler()
    yscaled = sc_y.fit_transform(ytrain)

    ## Training the model
    #  we'll need a class that handles this kind of prediction, and it is SVM
    from sklearn.svm import SVR
    # Creating an object of this class and defining the kernel, i e, (replace a dot product for
    # a non-linear function)
    regressor = SVR(kernel="rbf")

    # The training itself
    # The .fit method expects the "y value" in 1D array, so we use np.ravel to do this
    # the np.ravel, flattens the array
    regressor.fit(xscaled, np.ravel(yscaled))
    ## Predicting new results
    #  By the fact our data is scaled, the input predictions must be scaled too.
    #  This scaling method must be the one which were used in the matrix of features
    #  in our case here, (sc_x)
    #  regressor.predict(sc_x.transform([[6.5]])) # here, the predicted salary is in the scale that
    # was applied to y
    # therefore, instead of getting salary around 160K, we'll get a value around y range of values
    # So we have to reverse the scale of y to obtain the correct salary value and this method is
    # (inverse_transform). It's an object of "sc_y"
    ypred = sc_y.inverse_transform(regressor.predict(sc_x.transform(xtest)))
    score = r2_score(ytest,ypred)
    #print("R2 for SVR: ", score)
    return [score, "Suport Vector Regression"]


def DTR(xtrain, xtest, ytrain, ytest):

    ## Training the model
    regressor = DecisionTreeRegressor(random_state=0)
    regressor.fit(xtrain, ytrain)

    ## Predicting values
    ypred = regressor.predict(xtest)
    score = r2_score(ytest, ypred)
    #print("R2 for Decion Tree: ", score)
    return [score, "Decion Tree"]

def RFR(xtrain, xtest, ytrain, ytest):

    # Creating an object from this class
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(xtrain, ytrain)

    ## Predicting new values
    ypred = regressor.predict(xtest)
    score = r2_score(ytest,ypred)
    #print("R2 for Random Forest: ", score)
    return [score, "Random Forest Regression"]

def Model_selection(xtrain, xtest, ytrain, ytest):
    score_MLR = MLR(xtrain, xtest, ytrain, ytest)
    score_PlmR = PlmR(xtrain, xtest, ytrain, ytest)
    score_SVR = SVR(xtrain, xtest, ytrain, ytest)
    score_DTR = DTR(xtrain, xtest, ytrain, ytest)
    score_RFR = RFR(xtrain, xtest, ytrain, ytest)
    print("R2 for Multiple Linear Regression: ", score_MLR[0])
    print("R2 for Plynomial Regression: ", score_PlmR[0])
    print("R2 for SVR: ", score_SVR[0])
    print("R2 for Decion Tree: ", score_DTR[0])
    print("R2 for Random Forest: ", score_RFR[0])
    aux = np.array([score_MLR, score_PlmR, score_SVR, score_DTR, score_RFR])

    ##Creating a dataframe
    df = pd.DataFrame({"R_2":aux[:,0], "Regression_Type":aux[:,1]})
    best_regression = pd.DataFrame(df.max())
    print("The best model is ",best_regression)
    return

Model_selection(xtrain, xtest, ytrain, ytest)