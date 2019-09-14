# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing

## The dataset describes a store located in one of the most popular places in the south of France.
## So, a lot of people go into the store.

## And therefore the manager of the store noticed and calculated that on 
## average each customer goes and buys something to the store once a week.

## This dataset contains 7500 transactions of all the different customers
## that bought a basket of products in a whole week.

## Indeed the manage took it as the basis of its analysis because since each customer is going an 
## average once a week to the store then the transaction registered over a week is 
## quite representative of what customers want to buy.

## So based on all these 7500 transactions our machine learning model (apriori) is going to learn the
## the different associations it can make to actually understand the rules.
## Such as if customers buy this product then they're likely to buy this other set of products.

## Each line in the database (each observation) corresponds to a specific customer who bought a specific basket of product.
## For example, in line 2 the customer bought burgers, meatballs, and eggs

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset

## Mining Associations with Apriori: 
### Mine frequent itemsets, association rules or association hyperedges using the Apriori algorithm. 
### The Apriori algorithm employs level-wise search for frequent itemsets. 
### The implementation of Apriori used includes some improvements (e.g., a prefix tree and item sorting).

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Apriori - Algorithm
## Step 1: Set a mininum support and confidence
## Step 2: Take all the subsets in transactions having higher support than minimum support
## Step 3: Take all the rules of these subsets having higher confidence than minimum confidence
## Step 4: Sort the rules by decreasing lift

# Visualising the results
results = list(rules)
print("Visualising the results: ", results)