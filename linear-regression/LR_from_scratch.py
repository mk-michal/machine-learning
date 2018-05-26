from typing import Tuple, List


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def create_sample_data(n_points:int, variance = 1) -> Tuple[np.array, np.array]:
	'''Function that only creates some sample x and y data that are linear	
	Args:
	    n_points (int): number of observations that I want to display 
	    variance (int, optional): variance of the observations
	
	Returns:
	    Tuple[float, float]: Description
	'''
	x_values = np.array(variance * np.random.ranf(n_points))
	y_values = np.array(4 + 2* x_values + 0.2 * np.random.randn(n_points))

	return x_values, y_values


class LinearRegression():
	def __init__(self, x_values: np.array = None, y_values: np.array = None, n_observations = 100, variance = 1):

		self.coefficients = None
		self.scikit_coefficients = None
		self.res_error = 0
		if not x_values:
			self.x, self.y = create_sample_data(n_observations, variance)
		else:
			self.x = x_values
			self.y = y_values
			

	def plot_sample_data(self):
		'''Simply plots the given sample data
		'''
		plt.plot(self.x, self.y)
		return plt.show()


	def fit_LR(self):
		'''Method that calculates coefficients of linear regression'''
		
		beta_1 =  (np.sum(self.x * self.y) - len(self.x) * np.mean(self.x) * np.mean(self.y)) / \
		(np.sum(self.x**2) - len(self.x) * np.mean(self.x)**2)
		
		beta_0 = np.mean(self.y) - beta_1 * np.mean(self.x)

		self.coefficients = np.array((beta_0, beta_1))
		self.residual_error()


	def residual_error(self):
		'''Computes residual error of the regression'''
		for i, value in enumerate(self.y):
			self.res_error += (value - (self.coefficients[0] + self.coefficients[1] * self.x[i])) ** 2


	def LR_plot(self):
		'''Plotting results of linear regression using matplotlib'''

		plt.scatter(self.x,self.y)
		plt.plot(self.x, self.coefficients[0] + self.coefficients[1] * self.x)

		return plt.show()


	def MR_fit(self):
		"""My own construction of multiple regression model. 
		"""

		x = self.x.reshape(-1,1)
		y = self.y.reshape(-1,1)

		#defining vector of ones in order to join the vector with the x_values vector
		ones = np.ones(len(x)).reshape(-1,1)
		x_values = np.concatenate((ones, x), axis = 1)

		# computing coefficients by the formula given in 
		coefs = np.linalg.inv(x_values.T.dot(x_values)).dot(x_values.T).dot(y)
		plt.scatter(x, y)
		plt.plot(x,(coefs[0] + coefs[1] * x), lw = 2.5, label = 'fitting line')
		
		return plt.show()
		

	def LR_scikit_learn(self):
		'''Rerturns coefficients of given data and regression line by sci-kit learn'''
		
		regr = linear_model.LinearRegression()
		regr.fit(x_values, y_values)
		
		self.scikit_coefficients = regr.coef_
		

	def plot_scikit_learn(self):

		plt.scatter(self.x, self.y)
		plt.plot(self.x, self.scikit_coefficients[0] + self.scikit_coefficients[1] * self.x)

		return plt.show()
