from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"
# pio.renderers.default = 'png'


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    rand_data = np.random.normal(loc=10, scale=1, size=1000)
    est = UnivariateGaussian()
    est.fit(rand_data)
    # print(f"({est.mu_}, {est.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    tmp_est = UnivariateGaussian()
    exp_dist = np.zeros((2, 100))
    for i, n in enumerate(range(10, 1001, 10)):
        tmp_est.fit(rand_data[:n])
        exp_dist[:, i] = n, np.abs(est.mu_ - tmp_est.mu_)

    fig = px.scatter(x=[1, 2, 3, 5], y=[4, 3, 5, 6])
    fig.show()

    # # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
