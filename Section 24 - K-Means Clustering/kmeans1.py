'''Basically reassignment means that a point switched from one cluster to
 another. The green line is used to split the regions of each cluster, so that
 we can see which area belongs to which cluster. If a point stays in the same
 area of the previous iteration (a.k.a if it still belongs to the same cluster)
 there's no need to reassign it.'''
 
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
data = pd.read_csv('Mall_Customers.csv')
X = data.iloc[:, [3,4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss =[]
for i in range(1,11):
   kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter =300, n_init =10, random_state = 0)
   kmeans.fit(X)
   wcss.append(kmeans.inertia_)
   
plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("No of Clusters")
plt.ylabel("wcss")
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter =300, n_init = 10, random_state =0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1], s= 100, c = 'red', label ='cluster 1')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1], s= 100, c = 'green', label ='cluster 2')  
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1], s= 100, c = 'blue', label ='Target client')     
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1], s= 100, c = 'cyan', label ='cluster 4')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1], s= 100, c = 'magenta', label ='cluster5')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s= 300, c = 'yellow', label ='Centroid')
plt.title("Cluster of clients")
plt.xlabel("Annual income of clients")
plt.ylabel("Spending score in(1-100)")
plt.legend()
plt.show()

#init - Method for initialization, defaults to ‘k-means++’:  k-means++ is a method of
#selecting random, but no entirely random cluster starting points at the start of the model

#n_init - Number of time the k-means algorithm will be run with different centroid seeds.
#The final results will be the best output of n_init consecutive runs in terms of inertia.
# max_iter - Maximum number of iterations of the k-means algorithm for a single run.


#Both the codes would return same results. in [3,4] and [3:5] only column 3 and 4
#will be printed. But if you need more number of columns then you will have to use
#colon method.





    