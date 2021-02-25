## Importing the libraries
import numpy as np
import pandas as pd

## Loading the dataset
data = pd.read_csv("50_Startups.csv")

## Splitting Features and Classes
inputs = data.iloc[:,:-1].values
outputs = data.iloc[:,-1].values

## Encoding the categorial Data
#  These libraries are responsible for doing that
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(),[3])],remainder="passthrough")
inputs = np.array(ct.fit_transform(inputs))

## Now, we must split our data into train and test data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(inputs,outputs,test_size=0.2)

## Let's train the model in the tarining test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain,ytrain)

## Let's the test set results
ypred = regressor.predict(xtest)
print(ypred)

## Setting numpy to display any numerical value with two
#  decimals after comma:
np.set_printoptions(precision=2)
print(np.concatenate((ypred.reshape(len(ypred),1),ytest.reshape(len(ypred),1)),axis=1))