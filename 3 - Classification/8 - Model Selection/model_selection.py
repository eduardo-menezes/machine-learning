# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


## Loading the dataset
data = pd.read_csv("Data.csv")
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
## Now, we must split our data into train and test data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)

# Feature Scaling
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

def LogReg(xtrain, xtest, ytrain, ytest ):

    # Training the Logistic Regression model on the Training set
    classifier = LogisticRegression(random_state=0)
    classifier.fit(xtrain, ytrain)

    # Predicting the Test set results
    y_pred = classifier.predict(xtest)

    # Making the Confusion Matrix
    cm = confusion_matrix(ytest, y_pred)
    score = accuracy_score(ytest, y_pred)

    return [score, "Logistic Regression"]


def KNN(xtrain, xtest, ytrain, ytest ):

    classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    classifier.fit(xtrain, ytrain)

    # Predicting the Test set results
    y_pred = classifier.predict(xtest)
    cm = confusion_matrix(ytest, y_pred)
    score = accuracy_score(ytest, y_pred)
    return [score, "KNN"]

def SVM(xtrain, xtest, ytrain, ytest ):

    classifier = SVC(kernel="linear", random_state=0)
    classifier.fit(xtrain, ytrain)

    # Predicting the Test set results
    y_pred = classifier.predict(xtest)
    cm = confusion_matrix(ytest, y_pred)
    score = accuracy_score(ytest, y_pred)
    return [score, "Suport Vector Classification"]

def Kernel_SVM(xtrain, xtest, ytrain, ytest ):

    classifier = SVC(kernel="rbf", gamma="auto", random_state=0)
    classifier.fit(xtrain, ytrain)

    # Predicting the Test set results
    y_pred = classifier.predict(xtest)
    cm = confusion_matrix(ytest, y_pred)
    score = accuracy_score(ytest, y_pred)
    return [score, " Kernel Suport Vector Classification"]

def Naive_bayes(xtrain, xtest, ytrain, ytest ):

    classifier = GaussianNB()
    classifier.fit(xtrain, ytrain)

    # Predicting the Test set results
    y_pred = classifier.predict(xtest)
    cm = confusion_matrix(ytest, y_pred)
    score = accuracy_score(ytest, y_pred)
    return [score, " Naive Bayes Classification"]

def DTC(xtrain, xtest, ytrain, ytest):

    ## Training the model
    regressor = DecisionTreeClassifier(criterion="entropy",random_state=0)
    regressor.fit(xtrain, ytrain)

    ## Predicting values
    y_pred = regressor.predict(xtest)
    cm = confusion_matrix(ytest, y_pred)
    score = accuracy_score(ytest, y_pred)
    return [score, "Decion Tree Classification"]

def RFC(xtrain, xtest, ytrain, ytest):

    ## Training the model
    regressor = RandomForestClassifier(n_estimators=10, criterion="entropy", random_state=0)
    regressor.fit(xtrain, ytrain)

    ## Predicting values
    y_pred = regressor.predict(xtest)
    cm = confusion_matrix(ytest, y_pred)
    score = accuracy_score(ytest, y_pred)
    return [score, "Random Forest Classification"]

def Model_selection(xtrain, xtest, ytrain, ytest):

    score_LogReg = LogReg(xtrain, xtest, ytrain, ytest)
    score_KNN = KNN(xtrain, xtest, ytrain, ytest)
    score_SVM = SVM(xtrain, xtest, ytrain, ytest)
    score_Kernel_SVM = Kernel_SVM(xtrain, xtest, ytrain, ytest)
    score_Naive_bayes = Naive_bayes(xtrain, xtest, ytrain, ytest)
    score_DTC = DTC(xtrain, xtest, ytrain, ytest)
    score_RFC = RFC(xtrain, xtest, ytrain, ytest)

    # Creating an auxiliar var. to catch the function's values
    aux = np.array([score_LogReg, score_KNN, score_SVM, score_Kernel_SVM, score_Naive_bayes, score_DTC, score_RFC])

    ##Creating a dataframe
    df = pd.DataFrame({"Accuracy": aux[:,0], "Model":aux[:,1]})
    print(df)
    best_classification = pd.DataFrame(df.max(axis=0))
    print("The best model is ", best_classification)
    return

Model_selection(xtrain, xtest, ytrain, ytest)