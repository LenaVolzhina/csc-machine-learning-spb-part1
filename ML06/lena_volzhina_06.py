import csv
import numpy as np
import pandas as pd
from collections import Counter


def normalize_min_max(dataset, parameters=None):
	parameters = parameters or {}
	result = dataset.copy()
	for n_col in range(result.shape[1]):
		column = result[:, n_col]
		if n_col in parameters:
			# use precalculated avg and stdev
			minmax = parameters[n_col]
		else:
			minmax = column.min(), column.max()
			parameters[n_col] = minmax

		spread = (minmax[1] - minmax[0]) or 1
		result[:, n_col] = (column - minmax[0]) / spread
			
	return result, parameters


def save_submission(path, ids, answers):
	with open(path, 'w') as file:
		writer = csv.writer(file)
		writer.writerow(['id', 'label'])
		for i, answer in zip(ids, answers):
			writer.writerow([int(i.item()), answer.item()])


@np.vectorize
def sigmoid(x):
	return 1 / (1 + np.exp(-x))


@np.vectorize
def sigmoid1(x):
	return 1 - sigmoid(x)


class NeuralNetwork(object):
	def __init__(self, layer_shapes, m=3000, eps=1e-4, max_iter=100, alpha=5):
		self.shapes = layer_shapes
		self.biases = None
		self.weights = None
		self.alpha = alpha
		self.m, self.eps, self.max_iter = m, eps, max_iter
		self.n_layers = len(layer_shapes)

	def init_params(self):
		self.weights, self.biases = {}, {}
		for layer, shape in enumerate(zip(self.shapes[1:], self.shapes), 1):
			self.weights[layer] = np.random.normal(0, 1, shape)
			self.biases[layer] = np.random.normal()

	def calculate_activations(self, examples):
		# activations[l][j][k] = a_j^l for example k
		activations = {0: examples}
		for layer in range(1, self.n_layers):
			summators = self.weights[layer].dot(activations[layer - 1]) + self.biases[layer]
			activations[layer] = sigmoid(summators)
		return activations

	def calculate_errors(self, acts, ys):
		diff = lambda l: acts[l] * (1 - acts[l])
		
		max_layer = self.n_layers - 1
		errors = {max_layer: (ys - acts[max_layer]) * diff(max_layer)}
		for layer in range(max_layer - 1, 0, -1):
			errors[layer] = (self.weights[layer + 1].T.dot(errors[layer + 1])) * diff(layer)
		return errors

	def update_parameters(self, activations, errors):
		diffs = []
		for layer in range(1, self.n_layers):
			err, acts = errors[layer].mean(axis=1), activations[layer - 1].mean(axis=1)

			# update bias by mean gradient (having only one bias for every layer)
			bias_grad = self.alpha * err.mean()
			self.biases[layer] += bias_grad
			diffs.append(bias_grad)
			
			# update weight by mean gradient
			for j, ej in enumerate(err):
				weights_grad = self.alpha * acts * ej 
				diffs.append(weights_grad.mean())
				self.weights[layer][j] += weights_grad
		return np.array(diffs)

	def predict(self, examples):
		activations = self.calculate_activations(examples.T)
		return activations[self.n_layers - 1].T

	def fit(self, X, y):
		self.init_params()

		iter = 0
		while True:
			iter += 1
			# choose m examples to train
			idxs = np.random.choice(X.shape[0], self.m, replace=False)
			xs, ys = X[idxs], y[idxs]

			# calculate errors for each example and neuron
			activations = self.calculate_activations(xs.T)
			errors = self.calculate_errors(activations, ys)

			# update parameters using gradient
			diff = self.update_parameters(activations, errors)

			# check if reached termination
			if iter > self.max_iter:
				break


# calculate mutual information
def MI_simple(xs, ys):
	xs_probs = {v: len(xs[xs == v]) / len(xs) for v in set(xs)}
	ys_probs = {v: len(ys[ys == v]) / len(ys) for v in set(ys)}
	xs_ys_probs = {}
	for vy in ys_probs:
		for vx in xs_probs:
			idxs = (xs == vx) & (ys == vy)
			if idxs.any():
				xs_ys_probs[(vx, vy)] = len(xs[idxs]) / len(xs)

	i_xy = sum(p * np.log2(p / xs_probs[vx] / ys_probs[vy]) 
		       for (vx, vy), p in xs_ys_probs.items())

	return i_xy


# read datasets
dataset = pd.read_csv("learn.csv")
X, y = dataset.drop(['id', 'y'], axis=1).values, dataset['y'].values
X_test = pd.read_csv("test.csv")
X_test_idxs, X_test = X_test['id'].values, X_test.drop('id', axis=1).values
print("Learn dataset shape: {}, test dataset shape: {}".format(X.shape, X_test.shape))

# normalize datasets
X_norm, params = normalize_min_max(X)
X_test_norm, _ = normalize_min_max(X_test, params)

# find mutual information
X_round = np.round(X)   # X_norm_round = np.round(X_norm, 2) -- worse
mutual_info = np.array([MI_simple(X_round[:, f], y) 
	                    for f in range(X.shape[1])])
print("Calculated mutual information of each feature with y-labels")

# after several runs got multiple best:
best_runs = [(50, 61, 40), (50, 61, 60), (50, 61, 30), (50, 49, 20), (35, 65, 63), (25, 48, 25)]
for N, layer1_size, layer2_size in best_runs:
	print("Build neural network for {} most important features with inner layers of size {} and {}"
		  .format(N, layer1_size, layer2_size))

	# extract features with max mutual information with y
	good_features = np.argsort(mutual_info)[-N:]
	X_good = X_norm[:, good_features]
	X_test_good = X_test_norm[:, good_features]

	# create and learn neural network
	nn = NeuralNetwork([N, layer1_size, layer2_size, 1])
	nn.fit(X_good, y)

	# save submission
	y_predicted = nn.predict(X_test_good)
	save_submission(('final_submission_top_{}_two_layers_{}_{}.csv'
		             .format(N, layer1_size, layer2_size)), 
		             X_test_idxs, y_predicted)

