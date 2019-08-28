# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)
results_list = []
for i in range(0, len(results)):
    results_list.append([str(results[i][0]),str(results[i][1]),
                         str(results[i][2])])

#This is basically a bunch of indexing being done from the data provided in the 
#Apriori results. We are indexing the support, confidence, lift, and items to put into a table.