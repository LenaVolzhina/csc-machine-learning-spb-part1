import csv
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC




# TODO: try soft normalization (dividing by 2*std)
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


# try default SVC
clf = SVC(probability=True)
clf.fit(X_norm, y)
y_pred = clf.predict_proba(X_test_norm)

save_submission("default_svc_with_normalization.csv", X_test_idxs, y_pred[:, 1])
# 0.784 on public


def grid_search(parameters):
	x_train, x_test, y_train, y_test = train_test_split(
		X_norm, y, test_size=0.5, random_state=0)

	search = GridSearchCV(SVC(), parameters, cv=5, scoring=make_scorer(roc_auc_score))
	search.fit(x_train, y_train)

	print("Best parameters set found on development set:")
	print()
	print(search.best_params_)
	print()
	print("Grid scores on development set:")
	print()
	means = search.cv_results_['mean_test_score']
	stds = search.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, search.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r"
			  % (mean, std * 2, params))
	print()

	y_true, y_pred = y_test, search.predict(x_test)
	print('auc on test', roc_auc_score(y_true, y_pred))

	return search


best_clf = grid_search([{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
						 'C': [1, 10, 100, 1000]},
						{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}])
# best: {'gamma': 0.001, 'C': 10, 'kernel': 'rbf'}
# auc on test 0.742096195185
"""0.693 (+/-0.028) for {'gamma': 0.001, 'C': 1, 'kernel': 'rbf'}
0.673 (+/-0.033) for {'gamma': 0.0001, 'C': 1, 'kernel': 'rbf'}
0.726 (+/-0.041) for {'gamma': 0.001, 'C': 10, 'kernel': 'rbf'}
0.686 (+/-0.036) for {'gamma': 0.0001, 'C': 10, 'kernel': 'rbf'}
0.719 (+/-0.079) for {'gamma': 0.001, 'C': 100, 'kernel': 'rbf'}
0.691 (+/-0.032) for {'gamma': 0.0001, 'C': 100, 'kernel': 'rbf'}
0.694 (+/-0.053) for {'gamma': 0.001, 'C': 1000, 'kernel': 'rbf'}
0.674 (+/-0.068) for {'gamma': 0.0001, 'C': 1000, 'kernel': 'rbf'}
0.612 (+/-0.039) for {'C': 1, 'kernel': 'linear'}
0.598 (+/-0.060) for {'C': 10, 'kernel': 'linear'}
0.576 (+/-0.074) for {'C': 100, 'kernel': 'linear'}
0.576 (+/-0.074) for {'C': 1000, 'kernel': 'linear'}
"""

Cs, gammas = np.logspace(-10, 10, 11), np.logspace(-6, 0, 7)
degrees = np.linspace(1, 10, 1)
best_clf = grid_search([{'kernel': ['rbf'], 'gamma': gammas, 'C': Cs},
						{'kernel': ['linear'], 'C': Cs},
						{'kernel': ['poly'], 'C': Cs, 'degree': degrees}])
# best: {'gamma': 0.01, 'C': 1.0, 'kernel': 'rbf'}, on test 0.777
"""0.760 (+/-0.059) for {'gamma': 0.01, 'C': 1.0, 'kernel': 'rbf'}
0.739 (+/-0.060) for {'gamma': 0.10000000000000001, 'C': 1.0, 'kernel': 'rbf'}
0.719 (+/-0.079) for {'gamma': 0.001, 'C': 100.0, 'kernel': 'rbf'}
0.748 (+/-0.055) for {'gamma': 0.01, 'C': 100.0, 'kernel': 'rbf'}
0.740 (+/-0.055) for {'gamma': 0.10000000000000001, 'C': 100.0, 'kernel': 'rbf'}
0.748 (+/-0.055) for {'gamma': 0.01, 'C': 10000.0, 'kernel': 'rbf'}
0.740 (+/-0.055) for {'gamma': 0.10000000000000001, 'C': 10000.0, 'kernel': 'rbf'}
0.748 (+/-0.055) for {'gamma': 0.01, 'C': 1000000.0, 'kernel': 'rbf'}
0.740 (+/-0.055) for {'gamma': 0.10000000000000001, 'C': 1000000.0, 'kernel': 'rbf'}
0.748 (+/-0.055) for {'gamma': 0.01, 'C': 100000000.0, 'kernel': 'rbf'}
0.740 (+/-0.055) for {'gamma': 0.10000000000000001, 'C': 100000000.0, 'kernel': 'rbf'}
0.748 (+/-0.055) for {'gamma': 0.01, 'C': 10000000000.0, 'kernel': 'rbf'}
0.740 (+/-0.055) for {'gamma': 0.10000000000000001, 'C': 10000000000.0, 'kernel': 'rbf'}
0.730 (+/-0.065) for {'C': 100.0, 'kernel': 'poly'}
0.720 (+/-0.047) for {'C': 10000.0, 'kernel': 'poly'}
0.724 (+/-0.065) for {'C': 1000000.0, 'kernel': 'poly'}
0.724 (+/-0.065) for {'C': 100000000.0, 'kernel': 'poly'}
0.724 (+/-0.065) for {'C': 10000000000.0, 'kernel': 'poly'}
"""


def save_best(search):
	# best from gridsearch
	best_params = search.best_params_
	clf = SVC(**best_params, probability=True)
	clf.fit(X_norm, y)
	y_pred = clf.predict_proba(X_test_norm)

	save_submission(("gridsearch_{auc}_{kernel}_C={C}_gamma={gamma}.csv"
					 .format(auc=search.best_score_, **best_params)), X_test_idxs, y_pred[:, 1])


