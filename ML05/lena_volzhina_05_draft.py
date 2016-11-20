import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from collections import Counter


%cpaste
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
--


%cpaste
class Node(object):
	def __init__(self, **kwargs):
		# p = probability of label=c1
		self.is_leaf = kwargs.get('p') is not None
		if self.is_leaf:
			self.p = kwargs['p']
			self.pred = self.right = self.left = None
		else:
			# it's inner node
			self.predicate = kwargs['predicate']
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
--


%cpaste
def entropy(values):
	result = 0
	for class_size in Counter(values).values():
		f_i = class_size / len(values)
		result -= f_i * np.log(f_i)
	return result
--


%cpaste
class Tree(object):
	def fit(self, X, y):
		self.X, self.y = X, y
		self.min_size , self.threshold_step = 10, 10

		self.root = self._learn_node(np.arange(len(X)))

	def _find_predicate(self, indices):
		print("  find pred for {} indices".format(len(indices)))
		parent_entropy = entropy(self.y[indices])
		def information_gain(*children):
			result = parent_entropy
			for child_indices in children:
				result -= entropy(self.y[child_indices])
			return result

		max_IG, max_f, max_threshold_idx = None, None, None
		# iterate by features
		for f in range(self.X.shape[1]):
			values = self.X[indices, f]

			# try to find threshold by this feature
			sort_indices = indices[np.argsort(values)]
			for threshold in range(self.min_size, len(indices) - self.min_size + 1, self.threshold_step):
				IG = information_gain(sort_indices[:threshold], indices[threshold:])
				if not max_IG or IG > max_IG:
					max_IG, max_f, max_threshold_idx = IG, f, sort_indices[threshold]

		threshold = self.X[max_threshold_idx, max_f]
		print("opt for {} indices: x[{}] > {} (IG = {})".format(len(indices), max_f, threshold, max_IG))
		return lambda x: x[max_f] > threshold

	def _learn_node(self, indices):
		if len(set(self.y[indices])) == 1:
			# all objects are in one class
			most_common = self.y[indices][0]
			return Node(p=most_common)

		if len(indices) < 2 * self.min_size:
			# too small, must be leaf
			most_common, freq = Counter(self.y[indices]).most_common(1)[0]
			normed_freq = freq / len(indices)
			return Node(p=normed_freq if most_common == 1 else (1 - normed_freq))

		# find optimal predicate
		predicate = self._find_predicate(indices)
		true_values = np.apply_along_axis(predicate, 1, self.X[indices])
		
		# make child nodes
		left_indices, right_indices = indices[~true_values], indices[true_values]
		print('left, right:', len(left_indices), len(right_indices))
		if len(left_indices) * len(right_indices) == 0:
			# one of them is empty
			most_common, freq = Counter(self.y[indices]).most_common(1)[0]
			normed_freq = freq / len(indices)
			return Node(p=normed_freq if most_common == 1 else (1 - normed_freq))

		# have two full clild nodes
		print(len(indices), "started children")
		left, right = self._learn_node(left_indices), self._learn_node(right_indices)
		print(len(indices), "calculated children")
		return Node(predicate=predicate, left=left, right=right)
--





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

# y_predicted = np.random.randint(0, 2, y.shape)
# roc, auc = get_ROC_and_AUC(y_predicted, y)
# plot_roc(*roc, 'roc_random.png')
# OK, auc calculation seems to be correct

def try_drop_not_float():
	float_features = [f for f in range(X.shape[1])
	                  if all(isinstance(x, (float, int)) for x in X[:, f])]
	X, X_test = X[:, float_features], X_test[:, float_features]

	def plot_features(X, normed=False):
		plt.figure(figsize=(10, 80))
		n_features = X.shape[1]
		c0 = y == 0
		plt.subplots_adjust(hspace=.4)
		for f in range(n_features):
			print(f)
			values = X[:, f]
			plt.subplot(n_features, 1, f + 1)
			plt.title('f_{}'.format(f))
			plt.yscale('log')
			plt.hist((values[c0], values[~c0]), color=['r', 'g'], normed=normed)
		plt.savefig('features_hist{}.png'.format('_normed' if normed else ''))

	# plot_features(X)
	# plot_features(X, normed=True)


def try_to_predict(N=100):
	tree = Tree()
	tree.fit(X[:N], y[:N])
	y_predicted = np.array([tree.root.process(x) for x in X[:N]])
	_, auc = get_ROC_and_AUC(y_predicted, y)
	return auc
