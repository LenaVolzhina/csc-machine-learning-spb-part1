import csv
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
from itertools import combinations
from scipy.spatial.distance import pdist, cdist, squareform


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


def accuracy(a, b):
	assert len(a) == len(b)
	return sum(int(ai == bi) for ai, bi in zip(a, b)) / len(a)


def predict_for_test_set(k, distances, labels, test_idxs, cv=False):
	result = []
	for idx in test_idxs:
		point_distances = distances[:, idx]
		if cv:
			# need to use only labels from learn dataset
			argsorted = np.argsort(point_distances)
			argsorted = argsorted[~np.in1d(argsorted, test_idxs)]
			neighbours = argsorted[:k]
		else:
			neighbours = np.argsort(point_distances)[:k]

		if MOST_COMMON:
			cnt = Counter(labels[neighbours])
		else:
			cnt = Counter()
			for i, l in enumerate(labels[neighbours], 1):
				cnt[l] += 1 / i

		result.append(cnt.most_common(1)[0][0])
	return np.array(result)


def split_to_n_folds(X, y, n_folds):
	n = X.shape[0]
	fold_size = n // n_folds
	idxs = np.random.permutation(n)
	for i_start in range(0, n, fold_size):
		yield idxs[i_start:i_start + fold_size]


def cross_validate(X, y, k, distances, n_folds=4, verbose=False):
	accuracy_values = []

	for idxs in split_to_n_folds(X, y, n_folds):
		predicted = predict_for_test_set(k, distances, y, idxs, cv=True)
		accuracy_values.append(accuracy(predicted, y[idxs]))

	if verbose:
		print("Mean accuracy on test set = {:.4f}  +- {:.4f}"
			  .format(np.mean(accuracy_values), np.std(accuracy_values)))
	
	return accuracy_values


def save_submission(path, ids, answers):
	with open(path, 'w') as file:
		writer = csv.writer(file)
		writer.writerow(['id', 'label'])
		for i, answer in zip(ids, answers):
			writer.writerow([int(i.item()), int(answer.item())])


%cpaste
def calculate_cosine_distances(m1, m2):
	def cosine_dist(v1, v2):
		return 1 - np.dot(v1, v2) / np.sqrt(np.dot(v1, v1)) / np.sqrt(np.dot(v2, v2))

	result = np.apply_along_axis(
		lambda v1: np.apply_along_axis(lambda v2: cosine_dist(v1, v2), 1, m2), 
		1, m1
	)
	return result
--




dataset = pd.read_csv("learn.csv")
X, y = dataset.drop(['id', 'label'], axis=1).values, dataset['label'].values
X_test = pd.read_csv("test.csv")
X_test_idxs, X_test = X_test['id'].values, X_test.drop('id', axis=1).values
print("Learn dataset shape: {}, test dataset shape: {}".format(X.shape, X_test.shape))

# weight of each neighbour is ~ 1 / (its number in sorted by distance list)
MOST_COMMON = False

# normalize datasets
X_norm, params = normalize_min_max(X)
X_test_norm, _ = normalize_min_max(X_test, params)
print("Applied min-max normalization")

# calculate pairwise cosine distances
b = datetime.now()
print("Started calculation of pairwise cosine distances (can take ~2m)...")
distances_learn_learn = squareform(pdist(X_norm, 'cosine'))
distances_learn_test = cdist(X_norm, X_test_norm, 'cosine')
print("Calculated distances in {} seconds".format(int((datetime.now() - b).total_seconds())))
# 110 seconds

# cross-validate for different k
take_every = 3
cv_X, cv_y = X_norm[::take_every], y[::take_every]
cv_distances = distances_learn_learn[::take_every, ::take_every]
for k in [1, 2, 3, 5, 7, 9, 13, 17, 21, 25, 31, 37, 43]:
	print("For k={}".format(k), end="    ")
	cross_validate(cv_X, cv_y, k, cv_distances, 5, True)
"""
For k=1    Mean accuracy on test set = 0.9325  +- 0.0325
For k=2    Mean accuracy on test set = 0.9374  +- 0.0304
For k=3    Mean accuracy on test set = 0.9360  +- 0.0292
For k=5    Mean accuracy on test set = 0.8589  +- 0.1611
For k=7    Mean accuracy on test set = 0.9457  +- 0.0257
For k=9    Mean accuracy on test set = 0.9448  +- 0.0267
For k=13    Mean accuracy on test set = 0.9440  +- 0.0291
For k=17    Mean accuracy on test set = 0.8595  +- 0.1612
For k=21    Mean accuracy on test set = 0.9471  +- 0.0240
For k=25    Mean accuracy on test set = 0.9440  +- 0.0269
For k=31    Mean accuracy on test set = 0.9420  +- 0.0301
For k=37    Mean accuracy on test set = 0.9428  +- 0.0281
For k=43    Mean accuracy on test set = 0.9437  +- 0.0267
"""

predicted = predict_for_test_set(21, distances_learn_test, y, range(len(X_test)))
save_submission("k=21, cosine, weight=i ^ -1.csv", X_test_idxs, predicted)
# ON PUBLIC k = 21 gave 94.949%
