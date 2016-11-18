import csv
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations


def normalize(dataset, parameters=None):
	columns = dataset.columns[dataset.columns not in ['id', 'label']]
	parameters = parameters or {}
	for col in columns:
		if col in parameters:
			# use precalculated avg and stdev
			avg, stdev = parameters[col]
		else:
			avg, stdev = dataset[col].mean(), dataset[col].std() or 1
			parameters[col] = (avg, stdev)

		dataset[col] = (dataset[col] - avg) / stdev
			
	return dataset, parameters


def accuracy(a, b):
	assert len(a) == len(b)
	return sum(int(ai == bi) for ai, bi in zip(a, b)) / len(a)


%cpaste
class kNNClassifier(object):
	def __init__(self, k):
		self.k = k
		self.X = None
		self.y = None

	def fit(self, X, y):
		self.X = X
		self.y = y

	def find_closest(self, points):
		distances = defaultdict(list)
		# find k closes from learn dataset for each point
		for n_p1, p1 in self.X.iterrows():
			for n_p2, p2 in points.iterrows():
				# p1 from learn dataset, p2 from points to predict
				d = np.sqrt(np.mean(np.square(p1 - p2)))
				if len(distances[n_p2]) < self.k or any(d < dist for _, dist in distances[n_p2]):
					distances[n_p2].append((d, n_p1))
					distances[n_p2] = sorted(distances[n_p2])[:self.k]
		return distances

	def predict(self, points):
		closest = self.find_closest(points)
		result = []
		for point in points:
			classes = self.y[[idx for idx, _ in closest[point]]]
			predicted, _ = Counter(classes).most_common(1)[0]
			result.append(predicted)
		return pd.Series(result)
--


def split_to_n_folds(X, y, n_folds):
	n = X.shape[0]
	fold_size = n // n_folds
	indices = np.random.permutation(n)
	fold_indices = [indices[i_start:i_start + fold_size]
	                for i_start in range(0, n, fold_size)]

	for idxs in fold_indices:
		# (X_learn, y_learn), (X_test, y_test)
		yield ((X.drop(idxs).reset_index(), y.drop(idxs).reset_index()), 
			   (X.loc[idxs].reset_index(), y.loc[idxs].reset_index()))


def cross_validate(X, y, k, n_folds=10, verbose=False):
	accuracy_values, betas = [], []

	clf = kNNClassifier(k)
	for (X_L, y_L), (X_T, y_T) in split_to_n_folds(X, y, n_folds):
		clf.fit(X_L, y_L)
		predicted = clf.predict(X_T)
		accuracy_values.append(accuracy(predicted, y_T))

	if verbose:
		print("Mean accuracy on test set = {:.2f}  +- {:.2f}"
			  .format(np.mean(accuracy_values), np.std(accuracy_values)))
	
	return accuracy_values


def save_submission(path, ids, answers):
	with open(path, 'w') as file:
		writer = csv.writer(file)
		writer.writerow(['id', 'target'])
		for i, answer in zip(ids, answers):
			writer.writerow([int(i.item()), int(answer.item())])


dataset = pd.read_csv("learn.csv")
X, y = dataset.drop('label', axis=1), dataset['label']
test_X = pd.read_csv("test.csv")

# cross_validate(X, y, 1, verbose=True)
# Mean accuracy on test set =  111607226.40  +-  7626565383.15



X_shrink = X
beta = fit(X_shrink, y)
test_y = predict(test_X_shrink, beta)
# .format('_'.join(map(str, good_features)))
save_submission("submission_.csv".format(n), test_X[:,1], test_y)
