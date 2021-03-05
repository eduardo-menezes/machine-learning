## Eclat is used in business problems when we're interested
#  only in supports.
## Eclat is dealing with set of products and Apriori is dealing
#  with some rules

# Importing the libraries
import pandas as pd
import numpy as np

# Loading the data
df = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
## The apriori method expect a list as argument, so we need to transform our df
#  into a list
#Creating the values to count number of transactions(lines) and numbers of products(columns)
line = df.shape[0]
column = df.shape[1]

#Creating an empty list to receive the transactions and products
transactions = []
for i in range(0,line):
    transactions.append([str(df.values[i,j]) for j in range(0,column)])

## Training the apriori model
#  min_suport can be calculated as frequency of product per day (ex:3/day)
#  times the unit of the data transactions (here we have 7501 transactions per week), so the
#  unit is seven days (one week) divided by the total number of transactions
#  (3*7)/7501

#min_confidence in R is 0.8 but here no rules where returned. So divided by 2
# just a few rules where returned. Finally, dividing by 2 gain, good number of rules were
# returned (0.2)

# min_lift has a minimum value and it's 3. min_lift less than 3 does not return relevant rules.
# lift is the most relevant metric to measure a strength of a rule
from apyori import apriori

rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length = 2,
                max_length=2)

# To display the results directly from "rules" we must put "rules" into the list method
results = list(rules)


## Putting the results well organised into a Pandas DataFrame
# Here, we access the values with its indexes from "results" variable
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]

    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

## Displaying the results sorted by descending lifts
print(resultsinDataFrame.nlargest(n = 10, columns = 'Support'))