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
		print("Median error on learn set = {:.2f}  +- {:.2f}"
			  .format(np.median(learn_errors), np.std(learn_errors)))
	print("Median error on test set = {:.2f}  +- {:.2f}"
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
	combination_by_n_col = list(range(n_col))
	for degree in range(2, max_degree + 1):
		for columns in combinations(range(1, n_col), degree):
			new_columns.append(np.prod(X[:, columns], axis=1))
			combination_by_n_col.append(columns)

	return np.hstack((X, np.column_stack(new_columns))), combination_by_n_col


def save_submission(features, title):
	X_shrink = X[:, features]
	test_X_shrink = test_X[:, features]
	beta = fit(X_shrink, y)
	test_y = predict(test_X_shrink, beta)

	with open(title, 'w') as file:
		writer = csv.writer(file)
		writer.writerow(['id', 'target'])
		for i, answer in zip(test_X[:,1], test_y):
			writer.writerow([int(i.item()), answer.item()])



dataset = read_dataset("learn.csv", with_answers=True)
X, y = dataset[:, :-1], dataset[:, -1]
test_X = read_dataset("test.csv")

# print_stats(**cross_validate(X, y), verbose=True) 
# Mean error on test set =  111607226.40  +-  7626565383.15

# got it from forward selection
nice_features = [0, 25, 29, 37, 51, 92, 106, 110, 113, 117, 121, 133, 138, 147, 161, 176, 189]
X_extended, combs = extend(X[:, nice_features], max_degree=4)
test_X_extended, _ = extend(test_X[:, nice_features], max_degree=4)


def try_forward_selection():
	# after each good_features calculation run 
	# print_stats(**cross_validate(X[:, good_features], y))

	fw_result = list(sorted(test_forward_selection(X, y)))
	
	good_features = [f for error, (f,) in fw_result if error < 30]   # 166 of 202
	# Median error on test set = 37.72  +- 27.18
	# kaggle: public 3.7 o___O

	good_features = [f for error, (f,) in fw_result if error < 24.5]   # 23 of 202
	# Median error on test set = 5.27  +- 0.40
	# kaggle: public 4.87


	# ALWAYS TAKE CONSTANT
	fw_result = list(sorted(test_forward_selection(X, y, always_leave=[0])))
	
	good_features = list(set(f for error, fs in fw_result if error < 9.5 for f in fs))   # 74 of 202
	# Median error on test set = 6.04  +- 1.05

	good_features = list(set(f for error, fs in fw_result if error < 9.4 for f in fs))   # 14 of 202
	# Median error on test set = 4.76  +- 0.42
	# public 4.28
	# [0, 25, 29, 37, 51, 92, 106, 110, 113, 117, 121, 133, 138, 147, 161, 176, 189]


	# ALWAYS TAKE CONSTANT AND f24, f132, f146
	fw_result = list(sorted(test_forward_selection(X, y, always_leave=[0, 25, 133, 147])))
	
	good_features = list(set(f for error, fs in fw_result if error < 5.4 for f in fs))   # 62 of 202
	# Median error on test set = 5.64  +- 0.33

	good_features = list(set(f for error, fs in fw_result if error < 5.35 for f in fs))  # 21 of 202
	# Median error on test set = 4.97  +- 0.35

	good_features = list(set(f for error, fs in fw_result if error < 5.335 for f in fs))  # 13 of 202
	# Median error on test set = 4.91  +- 0.36


def try_extended():
	# stupid bruteforce
	X_extended, combs = extend(X, max_degree=2)
	ext_fw_result = list(sorted(test_forward_selection(X_extended, y)))		# about 10min
	ext_good_features = [f for error, (f,) in ext_fw_result]
	
	for n in range(5, 100, 5):
		good_featured = ext_good_features[:n]
	    print_stats(**cross_validate(X[:, good_features], y))

	print("good features from extended: {}".format(ext_good_features))
	print_stats(**cross_validate(X[:, ext_good_features], y))
	print("these features are multiplications of following: {}", [combs[f] for f in good_features])


def try_extended_after_forward_selection():
	# print_stats(**cross_validate(X_extended[:, good_features], y))
	fw_result = list(sorted(test_forward_selection(X_extended, y, always_leave=[0])))   # ~1m

	good_features = list(set(f for error, fs in fw_result if error < 9 for f in fs))   # 201 of 2516
	# Median error on test set = 11.07  +- 4.32
	
	good_features = list(set(f for error, fs in fw_result if error < 8.5 for f in fs))   # 34 of 2516
	# Median error on test set = 4.09  +- 0.46  

	good_features = list(set(f for error, fs in fw_result if error < 8.2 for f in fs))   # 18 of 2516
	# Median error on test set = 3.96  +- 0.36

	# [0, 1, 225, 228, 230, 518, 232, 522, 11, 13, 174, 527, 180, 182, 183, 185, 123, 415]
	print("these features are multiplications of following: ", 
		  [combs[f] if f > 202 else (f,) for f in sorted(good_features)])
	# these features are multiplications of following:  
	# [(0,), (1,), (11,), (13,), (123,), (174,), (180,), (182,), (183,), (185,), 
	# (1, 10, 15), (1, 11, 13), (1, 11, 15), (1, 12, 13), (4, 5, 10), (5, 11, 13), (5, 12, 13), (5, 13, 15)]

def try_correlations():
	corr = np.corrcoef(X)

	def get_min_correlated():
		result, corrs, not_used = [], [], list(range(X.shape[1]))
		while not_used:
			# find feature from not_used which has minimum correlation with already selected
			min_feature, min_corr = None, 1
			for f in not_used:
				f_corr = max(abs(corr[f, r]) for r in result) if result else 0
				if min_feature is None or f_corr < min_corr:
					min_feature, min_corr = f, f_corr
			result.append(min_feature)
			corrs.append(min_corr)
			not_used.remove(min_feature)
		return result, corrs

	min_correlated_result = get_min_correlated()

	for alpha in np.arange(0.3, 1, 0.05):
		min_correlated = [f for f, c in zip(fw_result, cs) if c < alpha]
		print("alpha: {:.2f}, n_features: {: 3d}".format(alpha, len(min_correlated)), end='    ')
		X_shrink = X[:, min_correlated]
		print_stats(**cross_validate(X_shrink, y))
	"""
	alpha: 0.30, n_features:   5    Median error on test set = 9.10  +- 0.48
	alpha: 0.35, n_features:   6    Median error on test set = 9.17  +- 0.47
	alpha: 0.40, n_features:   7    Median error on test set = 9.33  +- 0.69
	alpha: 0.45, n_features:   8    Median error on test set = 9.32  +- 0.58
	alpha: 0.50, n_features:   9    Median error on test set = 9.29  +- 0.72
	alpha: 0.55, n_features:  10    Median error on test set = 10.09  +- 30.05
	alpha: 0.60, n_features:  11    Median error on test set = 10.11  +- 25.86
	alpha: 0.65, n_features:  13    Median error on test set = 10.06  +- 22.43
	alpha: 0.70, n_features:  15    Median error on test set = 10.08  +- 35.32
	alpha: 0.75, n_features:  16    Median error on test set = 10.24  +- 33.50
	alpha: 0.80, n_features:  20    Median error on test set = 11.82  +- 27.81
	alpha: 0.85, n_features:  26    Median error on test set = 4537.22  +- 35011.83
	alpha: 0.90, n_features:  31    Median error on test set = 73.13  +- 20872.62
	alpha: 0.95, n_features:  40    Median error on test set = 1566.03  +- 18643.79

	"""

	min_correlated = [f for f, c in zip(*min_correlated_result) if c < 0.8]
	fw_result = list(sorted(test_forward_selection(X, y, only_features=min_correlated)))
	good_features = [f for error, (f,) in fw_result if error < 30]   # 17
	print_stats(**cross_validate(X[:, good_features], y))
	# Median error on test set = 9.57  +- 0.48
	# but on public 9.8 o__O

	min_correlated = [f for f, c in zip(*min_correlated_result) if c < 0.5]
	fw_result = list(sorted(test_forward_selection(X, y, only_features=min_correlated)))
	good_features = [f for error, (f,) in fw_result if error < 30]   # 18
	print_stats(**cross_validate(X[:, good_features], y))
	# Median error on test set = 8.81  +- 0.48
	# public: 8.75


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



# print_stats(**cross_validate(X[:, good_features], y))
save_submission(good_features, "submission_nice_features_extended_to_degree_4_max_error_8.2.csv", from_extended=True)
