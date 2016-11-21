import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from collections import Counter


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


def entropy(values):
	result = 0
	for class_size in Counter(values).values():
		f_i = class_size / len(values)
		result -= f_i * np.log2(f_i)
	return result


def get_predicate(is_numeric, feature, threshold):
	if is_numeric:
		return lambda x: x[feature] > threshold
	else:
		return lambda x: x[feature] == threshold


class Node(object):
	def __init__(self, **kwargs):
		# p = probability of label=c1
		self.is_leaf = kwargs.get('p') is not None
		self.size = kwargs['size']
		if self.is_leaf:
			self.p = kwargs['p']
			self.predicate = self.right = self.left = None
		else:
			# it's inner node
			self.predicate = get_predicate(*kwargs['predicate_params'])
			self.left, self.right = kwargs['left'], kwargs['right']
			self.p = None

	def process(self, x):
		if self.is_leaf:
			return self.p
		else:
			if self.predicate(x):
				return self.right.process(x)
			else:
				return self.left.process(x)


class Tree(object):
	def __init__(self, threshold_step=10, min_size=10):
		self.X, self.y = None, None
		self.root = None
		self.threshold_step, self.min_size = threshold_step, min_size

	def fit(self, X, y):
		self.X, self.y = X, y

		self.root = self._learn_node(np.arange(len(X)))

	def _find_predicate(self, indices):
		parent_entropy = entropy(self.y[indices])
		n_rows = len(indices)
		max_IG, max_predicate = None, (None, None, None)
		# iterate by features
		for f in range(self.X.shape[1]):
			# try to find threshold by this feature
			values = self.X[indices, f]
			is_numeric = all(isinstance(x, (float, int)) for x in values)
			sort_indices = indices[np.argsort(values)]
			for idx in sort_indices[self.min_size:n_rows - self.min_size + 1:self.threshold_step]:
				value = self.X[idx, f]
				# try to divide feature by this value:
				# not very fast, TODO
				is_true = (values > value) if is_numeric else (values == value)
				left, right = indices[~is_true], indices[is_true]
				
				if len(left) > self.min_size and len(right) > self.min_size:
					IG = (parent_entropy - 
						  len(left) / len(indices) * entropy(y[left]) -
						  len(right) / len(indices) * entropy(y[right]))
					if not max_IG or IG > max_IG:
						max_IG = IG
						max_predicate = is_numeric, f, value

		if max_IG and max_IG > 0:
			return max_predicate

	def _learn_node(self, indices):
		def simple_node():
			most_common, freq = Counter(self.y[indices]).most_common(1)[0]
			normed_freq = freq / len(indices)
			return Node(p=normed_freq if most_common == 1 else (1 - normed_freq), size=len(indices))

		if len(set(self.y[indices])) == 1:
			# all objects are in one class
			return simple_node()

		if len(indices) < 2 * self.min_size:
			# too small, must be leaf
			return simple_node()

		# find optimal predicate
		predicate_params = self._find_predicate(indices)
		if predicate_params is None:
			# none of possible predicates are profitable
			return simple_node()
		
		# make child nodes
		predicate = get_predicate(*predicate_params)
		true_values = np.apply_along_axis(predicate, 1, self.X[indices])
		left_indices, right_indices = indices[~true_values], indices[true_values]
		if len(left_indices) * len(right_indices) == 0:
			# one of them is empty
			return simple_node()

		# have two full clild nodes
		left, right = self._learn_node(left_indices), self._learn_node(right_indices)
		return Node(predicate_params=predicate_params, 
			        left=left, right=right, size=len(indices))


def save_submission(path, ids, answers):
	with open(path, 'w') as file:
		writer = csv.writer(file)
		writer.writerow(['id', 'label'])
		for i, answer in zip(ids, answers):
			writer.writerow([int(i.item()), int(answer.item())])

dataset = pd.read_csv("learn.csv")
X, y = dataset.drop(['id', 'label'], axis=1).values, dataset['label'].values
X_test = pd.read_csv("test.csv")
X_test_idxs, X_test = X_test['id'].values, X_test.drop('id', axis=1).values
print("Learn dataset shape: {}, test dataset shape: {}".format(X.shape, X_test.shape))

tree = Tree(threshold_step=10, min_size=10)
tree.fit(X, y)
y_predicted = np.array([tree.root.process(x) for x in X_test])
save_submission('submission_simple_all_features_10_10.csv', X_test_idxs, y_predicted)
