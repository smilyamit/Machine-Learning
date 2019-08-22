
#how Algo decides where to split data ?
#The decision tree will split the data based on homogeneity.
# It will split the values based on what values are similar in certain instances. 
#This is the entropy portion of the intuition.
#Information entropy is the average rate at which information is produced by
# a stochastic source of data.

# Decision Tree Regressor Template
# Note DTR is not a good model in 1D it is best in more dim eg 3-D 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Decision Tree Regressor to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)


# Predicting a new result
y_pred = regressor.predict([[6.5]])  # only one [] denotes vector

# Visualising the Regression results
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regressor)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()