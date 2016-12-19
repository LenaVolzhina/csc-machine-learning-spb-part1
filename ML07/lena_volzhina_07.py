import csv
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


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


# read datasets
dataset = pd.read_csv("learn.csv")
X, y = dataset.drop(['id', 'y'], axis=1).values, dataset['y'].values
X_test = pd.read_csv("test.csv")
X_test_idxs, X_test = X_test['id'].values, X_test.drop('id', axis=1).values
print("Learn dataset shape: {}, test dataset shape: {}".format(X.shape, X_test.shape))

# normalize datasets
X_norm, params = normalize_min_max(X)
X_test_norm, _ = normalize_min_max(X_test, params)


# parameters found from gridsearch
best_params = {'gamma': 0.01, 'C': 1.0, 'kernel': 'rbf'}
print("Using parameters", best_params)
clf = SVC(probability=True, **best_params)
clf.fit(X_norm, y)
y_pred = clf.predict_proba(X_test_norm)

save_submission("final_submission.csv", X_test_idxs, y_pred[:, 1])
