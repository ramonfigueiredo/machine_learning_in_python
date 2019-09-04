# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# The dataset is very small, just 10 lines, therefore we will not split the dataset into the Training set and Test set

# Feature Scaling
'''
Linear regression is going to take care of feature scaling for us.
Therefore we don't need to do feature scaling.
'''

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Predicting a new result with Linear Regression
lin_reg_pred = lin_reg.predict([[6.5]])
print("Predicting a new result (6.5) with Linear Regression = {}".format(lin_reg_pred))

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

for deg in range(2, 5): # deg = 2, 3, 4
	poly_reg = PolynomialFeatures(degree = deg)
	X_poly = poly_reg.fit_transform(X)
	poly_reg.fit(X_poly, y)
	lin_reg_2 = LinearRegression()
	lin_reg_2.fit(X_poly, y)

	# Predicting a new result with Polynomial Regression
	poly_reg_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
	print("Predicting a new result (6.5) with Polynomial Regression (degree = {}) = {}".format(deg, poly_reg_pred))

	# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
	X_grid = np.arange(min(X), max(X), 0.1)
	X_grid = X_grid.reshape((len(X_grid), 1))
	plt.scatter(X, y, color = 'red')
	plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
	plt.title('Truth or Bluff (Polynomial Regression, degree={})'.format(deg))
	plt.xlabel('Position level')
	plt.ylabel('Salary')
	plt.show()