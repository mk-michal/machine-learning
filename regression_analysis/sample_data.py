from typing import Tuple


import numpy as np


def create_linear_data(n_points:int, variance = 1) -> Tuple[np.array, np.array]:
	'''Function that only creates some sample x and y data that are linear	
	Args:
	    n_points (int): number of observations that I want to display 
	    variance (int, optional): variance of the observations
	
	Returns:
	    Tuple[float, float]: Description
	'''
	x_values = np.array(2 * variance * np.random.ranf(n_points))
	y_values = np.array(4 + 2* x_values + 0.2 * np.random.randn(n_points))

	return x_values, y_values


def create_polynomial_data(n_points: int = 100, variance = 1) -> Tuple[np.array, np.array]:
	'''Simply creates random data that are polynomial with some random noise'''

	x_values = np.array(2 * variance * np.random.ranf(n_points))
	y_values = np.array(4 + 6 * x_values ** 2 - 5 * x_values + 0.15 * np.random.randn(n_points))

	return x_values, y_values

