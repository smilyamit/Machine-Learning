
# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values  # InDependent Variable
y = dataset.iloc[:, 2].values    # Dependent Variable

# Fitting Linear regression to dataset, no need to do train, testdueto lessdata
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

#Fitting polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x) # this plonomial transformation can be done for diff matr
poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# Visualizing the linear regression results
plt.scatter(x,y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('real or bluff(Linear Regression model)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# Visualizing the polynomial regression results
#x_grid = np.arange(min(x), max(x), 0.1) # optional -just making more resolution
#x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('real or bluff(Linear Regression model)')
plt.xlabel('position')
plt.ylabel('salary')
plt.show()

# Visualizing a particular result in linear regression 
lin_reg.predict([[6.5]])  # [] -1D array [[], []] -2D aaray

# Visualizing a particular result in polynomial regression 
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))



