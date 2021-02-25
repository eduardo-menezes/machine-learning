## Here we'll develop an optimization algorthim to solve the linear problem
## The objective function is the Sum Squared Error and we'll minimize it
open("DataE.py")
open("Linear Regression.py")
open("estimates.py")

import DataE as dt
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import estimates as est


# Let's define the SSE function to monitor a decrement

def SSE(x,y,pbeta0,pbeta1,i):
    error = 0
    n = len(x)
    for i in range(n):
        error = (y[i] - (pbeta0 + pbeta1*x[i]))**2  ## caculates the error between predicted value and the real one

    error = (1/2*n) * (error) # calculates the expected error

    return error


# Now we define the gradient function to calcule the direction

def grad(x,y,pbeta0,pbeta1,i):
    n = len(x)
    gradbeta0= -2/n * (y[i] - pbeta0 - pbeta1*x[i])
    gradbeta1 = -2/n * (y[i] - pbeta0 - pbeta1 * x[i]) * x[i]
    gradien = np.array([gradbeta0, gradbeta1])
    return gradien

# Now we define the search direction (steepest Descent / gradient descent)

def steepestdescent(x, y, pbeta0, pbeta1, tol,iter,alpha):

    n = len(x) #the number of variables
    #  it's the step length aka learning rate / it can also be calculated with armijo tec.
    for i in range(iter):
        gradbeta0 = 0
        gradbeta1 = 0  #creating the parameters to accumulate the values
        f = SSE(x,y,pbeta0,pbeta1,i)

        for j in range(n):
            gradbeta0 = grad(x,y,pbeta0,pbeta1,j)[0] + gradbeta0
            gradbeta1 = grad(x,y,pbeta0,pbeta1,j)[1] + gradbeta1


        # if the norm of the gradient i less or equal than a tolerance, we break the loop
        # because it converges
        normbeta0 = np.linalg.norm(gradbeta0)
        normbeta1 = np.linalg.norm(gradbeta1)

        if normbeta0 <= tol and normbeta1 <= tol:
            print("norm less or equal than the tolerance")
            break
        # Here da search direction is the opposite from the gradient
        dbeta0 = -gradbeta0
        dbeta1 = -gradbeta1

        # the next point is
        pbeta0 = pbeta0 + alpha * dbeta0
        pbeta1 = pbeta1 + alpha * dbeta1

    print("Intecept",pbeta0, ",", "slope", pbeta1)

    return pbeta0,pbeta1

def lregression(x, y, beta0, beta1, tol,iter,alpha):

    # Here we obtain the predicted parameters
    bhat0,bhat1 = steepestdescent(x, y, beta0, beta1, tol, iter, alpha)

    #Here we calculate the predicted values
    predictedy = bhat0 + bhat1*x

    # Plotting
    plt.plot(x,predictedy)
    plt.scatter(x,y)
    plt.show()

    return predictedy,bhat0,bhat1

def modelassessment(x, y, beta0, beta1, tol,iter,alpha):

    predictedy,bhat0,bhat1 = lregression(x, y, beta0, beta1, tol, iter, alpha)
    # Now we're going to assess the model( first level)
    rsqrd = dt.pearson(x,y)
    print("rquadrado", rsqrd)

    #confidence Interval
    n = len(x)  # number of observations
    p = 1  # number of independent variables
    significance = 0.05
    quant = scipy.stats.t.ppf(1 - (significance / 2), n - p - 1)
    print("Critical value", quant)
    std = est.stdbeta1(y, predictedy,x)
    print("standard error", std)
    ci = [bhat1 - (quant*std), bhat1 + (quant*std)]
    print("Confidence interval", ci)

    # Hypothesis Testing on coefficients - F statistic
    SSE = np.sum((y - predictedy)**2)
    ybar = np.mean(y)
    SSR = np.sum((predictedy - ybar)**2)
    Fstat = (SSR/SSE)*(n-2)
    print("F statistic", Fstat)

    # Calculating the residual and the standard residuals
    residual = y - predictedy
    xbar = dt.mean(x)
    pii1 = [(item-xbar)**2 for item in x] # comprehension to avoid for loop
    pii2 = np.sum((x-xbar)**2)

    # calculates the parameter pii
    pii = (1/n) + (pii1/pii2)
    s = np.sqrt(dt.variance(x))

    # Here we have the normalized residual
    z = np.divide(residual,(np.multiply(s,(np.sqrt(1-pii)))))

    #Plotting stdResiduals x predicted values
    yup = 2*np.ones(n)
    ydown = -2*np.ones(n)
    plt.figure()
    plt.scatter(predictedy, z, color='gray')
    plt.plot(predictedy,yup)
    plt.plot(predictedy,ydown)
    plt.show()

    # Comprehension to collect the outliers that are out of [-2,2]
    outliersup = [item for item in z if item > 2]
    outliersup.sort(reverse=True)
    outliersdown = [item for item in z if item < -2]
    outliersdown.sort(reverse=False)
    print(outliersup,outliersdown)

    return residual, predictedy

''' exemplo 1
url = "C:/Users/eduar/PycharmProjects/pythonProject1/Anscombe_quartet_data.csv"
data = pd.read_csv(url)
x = data['x123'].values
y = data['y1'].values
'''

# exemplo 2

data = np.genfromtxt("bonds.txt",skip_header=1)
x = data[:,1]
y = data[:,2]
beta0 = 0
beta1 = 0
tol = 1e-5
iter = 100000
alpha = 0.001

modelassessment(x, y, beta0, beta1, tol, iter, alpha)

''' as we can see, the step length here is affecting the speed convergence, and the number of necessary iterations
    are too high. So we need to compute the step length and bracketing  '''
