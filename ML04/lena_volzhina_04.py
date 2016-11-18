import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



%cpaste
def calculate_LDA_coefs(X, y):
	classes = set(y)
	M = {c: X[y == c].mean(axis=0) for c in classes}
	sigmas = {c: np.cov(X[y == c].T) for c in classes}
	S1 = np.linalg.inv(sum(sigmas.values()))

	# (x - centroids[c]).T.dot(s).dot( (x - centroids[c]).T )
	# прямая p_c1(x) / p_c2(x) = 1 задается 
	# x.T * S^-1 * (M_c2 - M_c1) - 1/2 (M_c2 + M_c1).T * S^-1 * (M_c2 - M_c1) + log(p(c1) / p(c2)) = 0
	# x.T * A  - B = 0
	c1, c2 = 0, 1
	B = 1/2 * (M[c2] + M[c1]).T.dot(S1).dot(M[c2] - M[c1]) - np.log(len(X[y==c1]) / len(X[y==c2]))
	A = S1.dot(M[c2] - M[c1])

	return A, B

"""with bug:
(array([ -1.26906286e-02,   1.11129438e-02,  -5.85029240e-05,
         -1.97738122e-02,  -8.36600888e+00,   9.67253257e+00,
         -5.25233790e-04,   5.54040175e-04,  -6.55093967e-04,
          5.68986865e-03]), 6.1841485014961037)

without bug:
(array([ -1.40333619e-02,   3.66497825e-03,  -1.95688319e-05,
         -1.06526800e-02,  -5.30334838e+00,   4.99480683e+00,
         -2.06386416e-04,   5.15115416e-04,  -5.05110617e-04,
          3.33789722e-03]), 3.4915993346198313)



--


%cpaste
def use_LDA(points, A, B):
	result = []
	for p in points:
		result.append(p.T.dot(A) - B)
	return np.array(result)
--


%cpaste
def get_ROC(y_predicted, y, N=100):
	t, dt = min(y_predicted), (max(y_predicted) - min(y_predicted)) / N
	c0, c1 = 0, 1     # assuming c1 as positive
	n_c0, n_c1 = len(y[y == c0]), len(y[y == c1])
	result = []
	for _ in range(N):
		FP = len(y_predicted[(y_predicted > t) & (y == c0)])
		TP = len(y_predicted[(y_predicted > t) & (y == c1)])

		TPR = TP / n_c1      # sensitivity
		FPR = FP / (n_c0 + n_c1)      # 1 - specifity
		result.append((FPR, TPR))

		t += dt

	return result
--


def plot_roc(roc):
	xs, ys = [x for x, _ in roc], [y for _, y in roc]
	plt.figure()
	plt.plot(xs, ys)

	plt.xlabel('false positive rate')
	plt.ylabel('true positive rate')
	plt.savefig('roc')

def calculate_AUC(a, b):
	pass


def split_learn_test(X, y, learn_part=0.5):
	n, n_learn = X.shape[0], int(X.shape[0] * learn_part)
	indices = np.random.permutation(n)
	learn_idx, test_idx = indices[:n_learn], indices[n_learn:]

	X_L, X_T = X[learn_idx,:], X[test_idx,:]
	y_L, y_T = y[learn_idx,:], y[test_idx,:]

	return (X_L, y_L), (X_T, y_T)


def cross_validate(X, y, iterations_num=500):
	learn_errors, test_errors, params = [], [], []
	for _ in range(iterations_num):
		(X_L, y_L), (X_T, y_T) = split_learn_test(X, y)

		A, B = calculate_LDA_coefs(X_L, y_L)
		
		learn_errors.append(calculate_AUC(use_LDA(X_L, A, B), y_L))
		test_errors.append(calculate_AUC(use_LDA(X_T, A, B), y_T))
		params.append((A, B))

	return {'learn_errors': learn_errors, 'test_errors': test_errors, 'params': params}


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


def try_stupid_LDA()
	A, B = calculate_LDA_coefs(X, y)
	
	# try to find good threshold
	predicted = use_LDA(X, A, B)

	plt.figure()
	plt.hist(predicted[y==0], color='b', alpha=0.3)
	plt.hist(predicted[y==1], color='r', alpha=0.3)
	plt.savefig('classes_hist')

	predicted_test = use_LDA(X_test, A, B)

	def make_submission(threshold):
		result = np.ones(len(X_test))
		result[predicted_test < threshold] = 0
		save_submission("stupid_LDA_{}.csv".format(threshold), X_test_idxs, result)

	for threshold in [-3, -2, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1]:
		make_submission(threshold)

	# -2 is the best


def try_LDA_with_dataset_cleaning():
	classes = set(y)
	M = {c: X[y == c].mean(axis=0) for c in classes}
	sigmas = {c: np.cov(X[y == c].T)}
	
	def filter_c1_by_closeness(X, y):
		closer_to_c1 = np.linalg.norm(X - M[1], axis=1) < np.linalg.norm(X - M[0], axis=1)
		really_c1 = closer_to_c1 & (y == 1)
		return X[really_c1 | (y == 0)], y[really_c1 | (y == 0)]

	X_filter, y_filter = filter_c1_by_closeness(X, y)
	A, B = calculate_LDA_coefs(X_filter, y_filter)

	# try to find good threshold
	predicted = use_LDA(X_filter, A, B)

	plt.figure()
	plt.hist(predicted[y_filter==0], color='b', alpha=0.3)
	plt.hist(predicted[y_filter==1], color='r', alpha=0.3)
	plt.savefig('classes_hist_filtered')

	predicted_test = use_LDA(X_test, A, B)

	def make_submission(threshold):
		result = np.ones(len(X_test))
		result[predicted_test < threshold] = 0
		save_submission("LDA_with_filtered_c1_{}.csv".format(threshold), X_test_idxs, result)

	for threshold in [-3, -2, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1]:
		make_submission(threshold)

