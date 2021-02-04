### importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Salary.Data.csv')
x = dataset.iloc[:, :-1].values  # X is matrix of independent variable
y = dataset.iloc[:, 1].values     # Y is vector of dependent variable

# splitting the data into training set and test set
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state =0)

# Fitting simple linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
 
#predicting the test set result
y_pred = regressor.predict(x_test)


#Visualizing the training set result
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("salary vs experience(Training set)")
plt.xlabel("Experience")
plt.ylabel("salary")
plt.show()

#Visualizing the test set result
plt.scatter(x_test, y_test, color = 'green')
plt.plot(x_train, regressor.predict(x_train), color = 'red')
plt.title("salary vs experience(Test set)")
plt.xlabel("Experience")
plt.ylabel("salary")
plt.show()

######


