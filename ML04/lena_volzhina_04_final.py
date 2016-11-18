import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv


def calculate_LDA_coefs(X, y):
	classes = [0, 1]
	M = {c: X[y == c].mean(axis=0) for c in classes}
	sigmas = {c: np.cov(X[y == c].T) for c in classes}
	S1 = np.linalg.inv(sum(sigmas.values()))

	# прямая p_c1(x) / p_c2(x) = 1 задается 
	# x.T * S^-1 * (M_c2 - M_c1) - 1/2 (M_c2 + M_c1).T * S^-1 * (M_c2 - M_c1) + log(p(c1) / p(c2)) = 0
	# x.T * A  - B = 0
	c1, c2 = classes
	B = 1/2 * (M[c2] + M[c1]).T.dot(S1).dot(M[c2] - M[c1]) - np.log(len(X[y==c1]) / len(X[y==c2]))
	A = S1.dot(M[c2] - M[c1])

	return A, B


def use_LDA(points, A, B):
	result = []
	for p in points:
		result.append(p.T.dot(A) - B)
	return np.array(result)


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
	save_submission("simple_LDA_{}.csv".format(threshold), X_test_idxs, result)

#for threshold in [-3, -2.4, -2.3, -2.2, -2.1, -2, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1]:
# 	make_submission(threshold)

# threshold = -2.1 is the best
make_submission(-2.1)

