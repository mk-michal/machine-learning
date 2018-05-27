from typing import Tuple, List


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


from regression_analysis.sample_data import create_linear_data 


class SGD:
	def __init__(self, x_values = None, y_values = None,  n_observations = 100, variance = 1):
		self.coefficients = None

		if not x_values:
			self.x, self.y = create_linear_data(n_observations, variance)
		else:
			self.x = x_values
			self.y = y_values


	def create_X_Y(self) -> Tuple[np.array, np.array]:
		'''Method that create X matrix and y vector in the right format in order 
		to parse it later to different SGD algorithms
		'''

		x_values = self.x.reshape(-1,1)
		Y = self.y.reshape(-1,1)

		# add ones into X matrix
		ones = np.ones(len(x_values)).reshape(-1,1)
		X = np.concatenate((ones, x_values), axis = 1)
		return X, Y


	def batch_SGD(self):
		'''Computs batch gradient descent based on formula in book
		'Hands-on machine learning' on the page 117
		'''
		
		X, Y = self.create_X_Y()

		step_size = 0.1
		n_iteration = 1000
		min_error = 0.001
		new_coefficients = np.random.rand(2,1)
		
		self.coefficients = np.array(([1000],[1000]))		
		# in each iteration I compute gradient vector, substract it from coefficients
		# and when the coefficients stopps changing, I break the loop
		for i in range(n_iteration):
			error = 0
			for ii in range(len(self.coefficients)):
				error += (self.coefficients[ii] - new_coefficients[ii]) ** 2
			error = np.sqrt(error)

			if error > min_error:
				self.coefficients = new_coefficients
				grad = 2 / len(X) * X.T.dot(X.dot(self.coefficients) - Y)
				new_coefficients = self.coefficients - step_size * grad
			else:
				print(i)
				break
		print(self.coefficients)


	def stochastic_GD(self):
		'''Calculates the parameters using stochastic GD (hands-on page 122)'''
		X, Y = self.create_X_Y()

		n_epochs = 50
		# new_coefficients = np.random.rand(2,1)
		self.coefficients = np.random.rand(2,1)		


		for epoch in range(n_epochs):
			for i in range(len(X)):
				index = np.random.randint(len(X))
				X_index = X[index:index + 1]
				Y_index = Y[index:index + 1]

				grad = 2 / len(X_index) * X_index.T.dot(X_index.dot(self.coefficients) - Y_index)
				
				eta = self.descending_coeffitient(epoch * len(X) + i)
				self.coefficients = self.coefficients - eta * grad

		print(self.coefficients)

	
	@staticmethod
	def descending_coeffitient(value):
		eta = 5/(50 + value)
		return eta


	def scikit_SGD(self):
		'''Implementation of SGD from scikit learn'''
		sgd_scikit = linear_model.SGDRegressor(n_iter = 50, penalty = None, eta0 = 0.1)
		sgd_scikit.fit(self.x.reshape(-1,1), self.y.reshape(-1,1))
		
		return sgd_scikit.intercept_, sgd_scikit.coef_