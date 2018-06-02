from sklearn import datasets
from sklear.linear_model import LogisticRegression
import numpy as np

# example: iris dataset

def iris_dataset_example():
	'''Returns x and y values of iris dataset'''
	iris = datasets.load_iris()
	y_values = iris.target.astype(np.int)
	x_values = iris.data

	return x_values, y_values


class LogReg:
	def __init__(self, x_values y_values):
		self.x_values = x_values
		self.y_values = y_values

	def scikit_logreg(self):
		scikit_logreg = LogisticRegression()
		
