from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

pio.templates.default = "simple_white"

# for some reason that's the only way i get it to display
# pio.renderers.default = "svg"

# limit nparray display to 3 decimal places
np.set_printoptions(precision=3)


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    rand_data = np.random.normal(loc=10, scale=1, size=1000)
    est = UnivariateGaussian()
    est.fit(rand_data)
    print(f"({est.mu_:.3f}, {est.var_:.3f})")

    # Question 2 - Empirically showing sample mean is consistent
    tmp_est = UnivariateGaussian()
    exp_dist = np.zeros((2, 100))
    for i, n in enumerate(range(10, 1001, 10)):
        tmp_est.fit(rand_data[:n])
        exp_dist[:, i] = n, np.abs(10 - tmp_est.mu_)

    fig1 = px.scatter(x=exp_dist[0, :], y=exp_dist[1, :],
                      title="Abs. distance of estimated expectation from true expectation as a function of sample size",
                      labels={'x': 'Sample Size', 'y': 'Absolute distance'})
    fig1.update_layout(title_x=0.5)
    fig1.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = np.zeros((2, 1000))
    pdfs[0, :] = rand_data
    pdfs[1, :] = est.pdf(rand_data)

    fig2 = px.scatter(x=pdfs[0, :], y=pdfs[1, :],
                      title="PDF by sample value",
                      labels={'x': 'Sample value', 'y': 'PDF'})
    fig2.update_layout(title_x=0.5)
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model

    mu = [0, 0, 4, 0]
    cov = [[1, 0.2, 0, 0.5],
           [0.2, 2, 0, 0],
           [0, 0, 1, 0],
           [0.5, 0, 0, 1]]
    rand_data = np.random.multivariate_normal(mu, cov, 1000)
    esti = MultivariateGaussian()
    esti.fit(rand_data)
    print(esti.mu_)
    print(esti.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    likeli_matrix = np.zeros((200, 200))

    num_of_samples = 4
    # Print the progression percentage of creating the heatmap
    print_prog = False

    for i in range(200):
        for j in range(200):
            likeli_matrix[i, j] = MultivariateGaussian.log_likelihood(np.array([f1[i], 0, f3[j], 0]), np.array(cov),
                                                                      rand_data[:num_of_samples, :])

        if print_prog and (i % 10) == 0:
            print(f"{i / 2}% . . .")
    fig4 = px.imshow(likeli_matrix, labels=dict(x="f1", y="f3", color="Log-Likelihood"),
                     x=f1,
                     y=f3,
                     color_continuous_scale=px.colors.sequential.YlOrRd,
                     origin="lower",
                     title="Log likelihood as a function of expectation estimator parameters", )
    fig4.update_layout(title_x=0.5)
    fig4.show()

    # Question 6 - Maximum likelihood
    amax_tuple = np.unravel_index(np.argmax(likeli_matrix), likeli_matrix.shape)
    print(f"\nchecked over {num_of_samples} samples\n"
          f"\nmaximum log-likelihood: {np.amax(likeli_matrix):.3f}\n"
          f"argmax's are:\n"
          f"f1={f1[amax_tuple[1]]:.3f}  f3={f3[amax_tuple[0]]:.3f}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
