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
	errors, betas = [], []
	it = 0
	while it < iterations_num:
		(X_L, y_L), (X_T, y_T) = split_learn_test(X, y)

		try:
			beta = fit(X_L, y_L)
		except e:
			print("degenerate matrix :(", type(e), e)
			continue

		errors.append(RMSE(predict(X_T, beta), y_T))
		betas.append(beta)
		it += 1

	return errors, betas


def print_stats(errors, betas_list=None, verbose=False):
	if betas_list:
		beta_len = betas_list[0].shape[0]
		for n_col in range(beta_len):
			column_coefs = [betas.item(n_col) for betas in betas_list]
			mean, stdev = np.mean(column_coefs), np.std(column_coefs)

			if verbose:
				print("beta[{: 4d}] = {:.2f}  +- {:.2f}".format(n_col, mean, stdev))

	print("Mean error on test set = {:.2f}  +- {:.2f}"
		  .format(np.mean(errors), np.std(errors)))


def test_forward_selection(dataset, answers, n_cols=1, always_leave=None, verbose=False):
	always_leave = always_leave or []
	results = []
	for columns in combinations(range(dataset.shape[1]), n_cols):
		dataset_part = dataset[:,list(set(list(columns) + always_leave))]
		errors, betas = cross_validate(dataset_part, answers, iterations_num=100)
		results.append((np.mean(errors) + np.std(errors), columns))  # pessimistic
		if verbose:
			print(("Using only features {} got mean error {:.2f}  +- {:.2f} ")
				  .format(columns, np.mean(errors), np.std(errors)), flush=True)
	return results


def extend(X, max_degree):
	new_columns = []
	n_col = X.shape[1]
	combination_by_n_col = {}
	for degree in range(2, max_degree + 1):
		for i, columns in enumerate(combinations(range(1, n_col), degree)):
			new_columns.append(np.prod(X[:, columns], axis=1))
			combination_by_n_col[n_col + i] = columns

	return np.hstack((X, np.column_stack(new_columns))), combination_by_n_col


def save_submission(path, ids, answers):
	with open(path, 'w') as file:
		writer = csv.writer(file)
		writer.writerow(['id', 'target'])
		for i, answer in zip(ids, answers):
			writer.writerow([int(i.item()), answer.item()])


dataset = read_dataset("learn.csv", with_answers=True)
X, y = dataset[:, :-1], dataset[:, -1]
test_X = read_dataset("test.csv")

# print_stats(*cross_validate(X, y), verbose=True) 
# Mean error on test set =  111607226.40  +-  7626565383.15

# _ = test_forward_selection(X, y, verbose=True)	# Mean error about 1, lowest for f24 (0.69)

# X_extended, _ = extend(X, max_degree=2)
# _ = test_forward_selection(X_extended, y, verbose=True)	# Mean error not lower than for X: [25, 226 (0x25)]


# take only first 20% with best error
fs = list(sorted(test_forward_selection(X, y)))
good_features = [f for error, (f,) in fs]
error = RMSE(predict(X[:, good_features], fit(X[:, good_features], y)), y)
print("good features: {}\n overall RMSE: {}".format(good_features, error))    # 4.03


def try_extended():
	X_extended, combs = extend(X, max_degree=2)
	ext_fs = list(sorted(test_forward_selection(X_extended, y)))		# about 10min
	ext_good_features = [f for error, (f,) in ext_fs]
	#ext_good_features = [f for f in ext_good_features if f in combs and 0 not in combs[f]]
	"""
	for n in range(170, 370, 10):  # [10, 20, 30, 40, 50, 60, 70, 120, 170, 270, 370, 470, 600, 800]
	    print(n, RMSE(predict(X_extended[:, ext_good_features[:n]], fit(X_extended[:, ext_good_features[:n]], y)), y))
	"""

	n = 330
	X_shrink = X_extended[:, ext_good_features[:n]]
	beta = fit(X_shrink, y)
	ext_error = RMSE(predict(X_shrink, beta), y)
	print("good features from extended: {}\n overall RMSE: {}".format(ext_good_features, ext_error)
	print("these features are multiplications of following: {}", [combs[f] for f in ext_good_features])
	



X_shrink = X_extended[:, ext_good_features[:n]]
beta = fit(X_shrink, y)
test_X_shrink = extend(test_X, max_degree=2)[0][:, ext_good_features[:n]]
test_y = predict(test_X_shrink, beta)
# .format('_'.join(map(str, good_features)))
save_submission("submission_f_top_{}_from_extended_with_degree_2.csv".format(n), test_X[:,1], test_y)
