import csv
import numpy as np
import pandas as pd
from collections import Counter

%cpaste

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


def get_ROC_and_AUC(y_predicted, y):
	c0, c1 = 0, 1 	# classes values
	fpr, tpr, auc = [0], [0], 0
	cnt = Counter(y)
	m0, m1 = cnt[c0], cnt[c1]
	for yi in y[np.argsort(y_predicted)[::-1]]:
		if yi == c0:
			fpr.append(fpr[-1] + 1 / m0)
			tpr.append(tpr[-1])
			auc += 1 / m0 * tpr[-1]
		else:
			fpr.append(fpr[-1])
			tpr.append(tpr[-1] + 1 / m1)
	return (fpr, tpr), auc


def plot_roc(xs, ys, filename):
	plt.figure()
	plt.plot(xs, ys)

	plt.xlabel('false positive rate')
	plt.ylabel('true positive rate')
	plt.xlim((0, 1))
	plt.ylim((0, 1))
	plt.savefig(filename)


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

--

%cpaste
class NeuralNetwork(object):
	def __init__(self, layer_shapes, m=3000, eps=1e-3):
		self.shapes = layer_shapes
		self.biases = None
		self.weights = None
		self.m, self.eps = m, eps
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
			# print(layer, errors[layer].shape, activations[layer - 1].shape)
			err, acts = errors[layer].mean(axis=1), activations[layer - 1].mean(axis=1)

			# update bias by mean gradient (having only one bias for every layer)
			bias_grad = err.mean()
			self.biases[layer] += bias_grad
			diffs.append(bias_grad)
			
			# update weight by mean gradient
			for j, ej in enumerate(err):
				weights_grad = acts * ej 
				diffs.append(weights_grad.mean())
				self.weights[layer][j] += weights_grad
		return np.array(diffs)

	def predict(self, examples):
		activations = self.calculate_activations(examples.T)
		return activations[self.n_layers - 1].T

	def fit(self, X, y):
		self.init_params()

		while True:
			# choose m examples to train
			idxs = np.random.choice(X.shape[0], self.m, replace=False)
			xs, ys = X[idxs], y[idxs]

			# calculate errors for each example and neuron
			activations = self.calculate_activations(xs.T)
			errors = self.calculate_errors(activations, ys)

			# update parameters using gradient
			diff = self.update_parameters(activations, errors)

			# check if reached termination
			mean_diff = abs(diff).mean()
			#if input('AUC = {:.4f}. Press Enter to continue, any other key to stop '
			#	     .format(auc(self.predict(xs), ys, reorder=True))):
			print(mean_diff)
			if mean_diff < self.eps:
				break
--


#nn = NeuralNetwork([924, 300, 50, 1])
#nn.fit(X_norm, y)


#nn.init_params()
#activations = nn.calculate_activations(X[:100].T)
#errors = nn.calculate_errors(activations, y[:100])
#diff = nn.update_parameters(activations, errors)


np.random.seed(0)
dataset = pd.read_csv("learn.csv")
X, y = dataset.drop(['id', 'y'], axis=1).values, dataset['y'].values
X_test = pd.read_csv("test.csv")
X_test_idxs, X_test = X_test['id'].values, X_test.drop('id', axis=1).values
print("Learn dataset shape: {}, test dataset shape: {}".format(X.shape, X_test.shape))

# normalize datasets
X_norm, params = normalize_min_max(X)
X_test_norm, _ = normalize_min_max(X_test, params)


%cpaste
def MI_simple(xs, ys):
	xs_probs = {v: len(xs[xs == v]) / len(xs) for v in set(xs)}
	ys_probs = {v: len(ys[ys == v]) / len(ys) for v in set(ys)}
	xs_ys_probs = {}
	for vy in ys_probs:
		for vx in xs_probs:
			idxs = (xs == vx) & (ys == vy)
			if idxs.any():
				xs_ys_probs[(vx, vy)] = len(xs[idxs]) / len(xs)

	#h_x = -sum(p * np.log2(p) for p in xs_probs.values())
	#h_y = -sum(p * np.log2(p) for p in ys_probs.values())
	#h_xy = -sum(p * np.log2(p) 
	#	       for (vx, vy), p in xs_ys_probs.items())
	i_xy = sum(p * np.log2(p / xs_probs[vx] / ys_probs[vy]) 
		       for (vx, vy), p in xs_ys_probs.items())

	return i_xy
--

X_norm_round = np.round(X_norm, 2)
X_round = np.round(X)
mutual_info = np.array([MI_simple(X_round[:, f], y) 
	                    for f in range(X.shape[1])])
good_features = np.argsort(mutual_info)[-10:]


# round X: [230,  69, 122, 112, 157, 123, 160, 354, 662, 665]


X_good = X_norm[:, good_features]

nn = NeuralNetwork([10, 7, 5, 1])
nn.fit(X_good, y)

y_predicted = nn.predict(X_test_norm[:, good_features])
save_submission('submission_X_round_top_10_1e-3_7_5.csv', X_test_idxs, y_predict)











%cpaste
def entropy_stupid(values):
	result = 0
	for v in set(values):
		p = len(values[values == v]) / len(values)
		result -= p * np.log2(p)
	return result
--


%cpaste
def entropy1(values, step=0.01):
	# only for normed to [0,1] datasets
	n = len(values)
	result = 0
	for start in np.arange(0, 1, step):
		idxs = (start <= values) & (values <= start + step)
		if idxs.any():
			p = len(values[idxs]) / n
			result -= p * np.log2(p)
	return result
--

%cpaste
def entropy2(values1, values2, step=0.01):
	n = len(values1)
	result = 0
	for start1 in np.arange(0, 1, step):
		idxs1 = (start1 <= values1) & (values1 <= start1 + step)
		if not idxs1.any():
			continue

		for start2 in np.arange(0, 1, step):
			idxs = (idxs1 & (start2 <= values2) & (values2 <= start2 + step))
			if idxs.any():
				p = len(values1[idxs]) / n 
				result -= p * np.log2(p)
	return result
--




from sklearn.metrics import auc
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neural_network import MLPClassifier

# [282 422 285 751 483 757 767 371 375 380 381 472 361 772 463 781 782 395
# 408 412 433 432 423 468 747 359 744 581 299 578 701 575 714 866 805 307
# 312 556 860 551 538 722 726 532 729 346 733 526 874  10]


clf = MLPClassifier()#(solver='sgd', hidden_layer_sizes=(N // 2, N // 5), random_state=1)
clf.fit(X_norm[:, good_features], y)
auc(clf.predict(X_norm[:, good_features]), y, reorder=True)

