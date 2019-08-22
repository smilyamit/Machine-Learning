'''seems there is no predict method in HC? so if I got a new dataset and want to
apply the trained cluster model to it, how should I do this? any sample code
you can share?'''

#This is correct. HC is used to find groups or clusters that have not been already
#established. It is considered an unsupervised learnign model. In order to predict 
#the groups of new data, we simply need to add the data to the analysis.

# Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
Data = pd.read_csv("Mall_Customers.csv")
X = Data.iloc[:, [3,4]].values


# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward')) # ward minimizes the variance of the clusters being merged.
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = 'ward')
y_hc = hc.fit_predict(X) # y_hc:It is the vector that tells for each customer which cluster the customer belongs to

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

''' Ward's method is a criterion applied in hierarchical cluster
analysis. Ward's minimum variance method is a special case of the objective function 
approach originally presented by Joe H. Ward, Jr.[1] Ward suggested a general 
agglomerative hierarchical clustering procedure, where the criterion for choosing 
the pair of clusters to merge at each step is based on the optimal value of an objective
 function. This objective function could be "any function that reflects the investigator's 
 purpose." Many of the standard clustering procedures are contained in this very general 
 class. To illustrate the procedure, Ward used the example where the objective function is
 the error sum of squares, and this example is known as Ward's method or more precisely Ward's
 minimum variance method.'''
 
 
 