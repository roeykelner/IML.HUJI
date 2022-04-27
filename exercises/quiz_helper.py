from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
from IMLearn.learners.classifiers import GaussianNaiveBayes, LDA, Perceptron
import numpy as np

# Ex 1
# x = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
#               -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])

# model = UnivariateGaussian()
# model.fit(x)
# print(model.log_likelihood(1, 1, x))
# print(model.log_likelihood(10, 1, x))

# print(np.var(np.array([0.917, 0.166, -0.03, 0.463])))

train_y = np.array([0, 0, 1, 1, 1, 1, 2, 2])
train_X = np.array([0, 1, 2, 3, 4, 5, 6, 7])

naive = GaussianNaiveBayes()
naive.fit(train_X, train_y)
print(f"Naive pi : {naive.pi_}")
print(f"Naive mu_ : {naive.mu_}")

S = np.array([[1, 1, 0], [1, 2, 0], [2, 3, 1],
             [2, 4, 1], [3, 3, 1], [3, 4, 1]])
x = S[:, :2]
y = S[:, 2]
print(f"X= {x}")
print(f"y= {y}")
naive = GaussianNaiveBayes()
naive.fit(x, y)
print(f"Var: {naive.vars_}")

x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
y = np.array([0, 0, 1, 1, 1, 1, 2, 2])
naive.fit(x, y)
print(f"Pois mu: { naive.mu_}")

