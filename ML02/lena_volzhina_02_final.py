import csv
import numpy as np
from itertools import combinations


def read_dataset(path, with_answers=False):
	X = []
	with open(path) as file:
		reader = csv.reader(file)
		header = False
		for line in reader:
			if not header:
				header = True
				continue
			X.append([float(v) for v in line])

	def normalized(dataset):
		col_num = len(dataset[0]) + (-1 if with_answers else 0)
		for n_col in range(1, col_num):
			values = [dataset[n_row][n_col] for n_row in range(len(dataset))]
			
			avg = np.mean(values)
			stdev = np.std(values) or 1
			process = lambda x: (x - avg) / stdev
		
			for n_row in range(len(dataset)):
				dataset[n_row][n_col] = process(dataset[n_row][n_col])

		return dataset
	X = [[1] + row for row in normalized(X)]
	return np.matrix(X)


def fit(X, y):
	return (X.T * X).I * X.T * y


def predict(X, beta):
	return X * beta


def RMSE(a, b):
	return np.sqrt(np.mean(np.square(a - b)))


def split_learn_test(X, y, learn_part=0.5):
	n, n_learn = X.shape[0], int(X.shape[0] * learn_part)
	indices = np.random.permutation(n)
	learn_idx, test_idx = indices[:n_learn], indices[n_learn:]

	X_L, X_T = X[learn_idx,:], X[test_idx,:]
	y_L, y_T = y[learn_idx,:], y[test_idx,:]

	return (X_L, y_L), (X_T, y_T)


def cross_validate(X, y, iterations_num=500):
	learn_errors, test_errors, betas = [], [], []
	it = 0
	while it < iterations_num:
		(X_L, y_L), (X_T, y_T) = split_learn_test(X, y)

		try:
			beta = fit(X_L, y_L)
		except e:
			print("degenerate matrix :(", type(e), e)
			continue

		learn_errors.append(RMSE(predict(X_L, beta), y_L))
		test_errors.append(RMSE(predict(X_T, beta), y_T))
		betas.append(beta)
		it += 1

	return {'learn_errors': learn_errors, 'test_errors': test_errors, 'betas_list': betas}


def print_stats(test_errors, learn_errors, betas_list=None, verbose=False):
	if betas_list:
		beta_len = betas_list[0].shape[0]
		for n_col in range(beta_len):
			column_coefs = [betas.item(n_col) for betas in betas_list]
			mean, stdev = np.mean(column_coefs), np.std(column_coefs)

			if verbose:
				print("beta[{: 4d}] = {:.2f}  +- {:.2f}".format(n_col, mean, stdev))

	if verbose:
		print("Median rmse on learn set = {:.2f}  +- {:.2f}"
			  .format(np.median(learn_errors), np.std(learn_errors)))
	print("Median rmse on test set = {:.2f}  +- {:.2f}"
		  .format(np.median(test_errors), np.std(test_errors)))


def test_forward_selection(dataset, answers, n_cols=1, always_leave=None, verbose=False,
	                       only_features=None):
	always_leave = always_leave or []
	variable_columns = set(only_features or range(dataset.shape[1])) - set(always_leave)
	results = []
	for columns in combinations(variable_columns, n_cols):
		columns = list(sorted(set(list(columns) + always_leave)))
		dataset_part = dataset[:,columns]
		cv_result = cross_validate(dataset_part, answers, iterations_num=100)
		errors, betas = cv_result['test_errors'], cv_result['betas_list']
		results.append((np.mean(errors) + np.std(errors), columns))  # pessimistic
		if verbose:
			print(("Using only features {} got mean error {:.2f}  +- {:.2f} ")
				  .format(columns, np.mean(errors), np.std(errors)), flush=True)
	return results


def extend(X, max_degree):
	new_columns = []
	n_col = X.shape[1]
	combination_by_n_col = [(col,) for col in range(n_col)]
	for degree in range(2, max_degree + 1):
		for columns in combinations(range(1, n_col), degree):
			new_columns.append(np.prod(X[:, columns], axis=1))
			combination_by_n_col.append(columns)

	return np.hstack((X, np.column_stack(new_columns))), combination_by_n_col


def save_submission(features, title, from_extended=False):
	if not from_extended:
		X_shrink, test_X_shrink = X[:, features], test_X[:, features] 
	else:
		X_shrink, test_X_shrink = X_extended[:, features], test_X_extended[:, features] 
	beta = fit(X_shrink, y)
	test_y = predict(test_X_shrink, beta)

	with open(title, 'w') as file:
		writer = csv.writer(file)
		writer.writerow(['id', 'target'])
		for i, answer in zip(test_X[:,1], test_y):
			writer.writerow([int(i.item()), answer.item()])


#------------------------------------
# READ DATA

dataset = read_dataset("learn.csv", with_answers=True)
X, y = dataset[:, :-1], dataset[:, -1]
test_X = read_dataset("test.csv")

# make it True to make full calculations (can take ~2 min). 
# if True, result can vary because of random CV
DEBUG = False

if DEBUG:
	print_stats(verbose=True, **cross_validate(X, y)) 
	# Median rmse on learn set = 553.84  +- 24831.26
	# Median rmse on test set = 4062947.97  +- 679392577.53


#-------------------------------------
# FORWARD SELECTION

if DEBUG:
	fw_result = list(sorted(test_forward_selection(X, y, always_leave=[0])))
	nice_features = list(set(f for error, fs in fw_result if error < 9.4 for f in fs))   # 14 of 202
	
	print("\nCalculated error for each feature combined with constant, took ~15 best of them")
	print_stats(**cross_validate(X[:, nice_features], y))
	# Median rmse on test set = 4.76  +- 0.42
	# RMSE on public leaderboard 4.28
else:
	nice_features = [0, 25, 29, 37, 51, 92, 106, 110, 113, 117, 121, 133, 138, 147, 161, 176, 189]



#--------------------------------------
# EXTEND X with multiplications of it's columns to degree=4

X_extended, combs = extend(X[:, nice_features], max_degree=4)
test_X_extended, _ = extend(test_X[:, nice_features], max_degree=4)

if DEBUG:
	# takes about 1 min to calculate
	fw_result = list(sorted(test_forward_selection(X_extended, y, always_leave=[0])))
	
	good_features = list(set(f for error, fs in fw_result if error < 8.2 for f in fs))   # 18 of 2516
	# Median error on test set = 3.96  +- 0.36
	# [0, 1, 225, 228, 230, 518, 232, 522, 11, 13, 174, 527, 180, 182, 183, 185, 123, 415]
	
	print("\nExtended dataset with multiplications of it's columns, then repeated previous step\n"
		  "Took ~20 best of all columns (old and new)")
	print("These features are multiplications of following:", 
		  [combs[f] for f in sorted(good_features)])
	# these features are multiplications of following:  
	# [(0,), (1,), (11,), (13,), (123,), (174,), (180,), (182,), (183,), (185,), 
	# (1, 10, 15), (1, 11, 13), (1, 11, 15), (1, 12, 13), (4, 5, 10), (5, 11, 13), (5, 12, 13), (5, 13, 15)]
else:
	good_features = [0, 1, 225, 228, 230, 518, 232, 522, 11, 13, 174, 527, 180, 182, 183, 185, 123, 415]


print_stats(**cross_validate(X_extended[:, good_features], y))
save_submission(good_features, "submission_nice_features_extended_to_degree_4_max_error_8.2.csv", from_extended=True)
