from __future__ import annotations
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    def f(x):
        return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    epsilon = np.random.normal(0, noise, n_samples)
    X = np.linspace(-1.2, 2, n_samples)
    Y_noiseless = f(X)
    Y_noise = Y_noiseless + epsilon
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(Y_noise), 2 / 3)
    train_X, train_y, test_X, test_y = np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=X, y=Y_noiseless, mode="lines", name="true"))
    fig1.add_trace(go.Scatter(x=train_X[:, 0], y=train_y, mode="markers", name="train"))
    fig1.add_trace(go.Scatter(x=test_X[:, 0], y=test_y, mode="markers", name="test"))
    fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors, validation_errors = [], []
    for deg in range(0, 11):
        tr_e, val_e = cross_validate(PolynomialFitting(deg), train_X, train_y, mean_square_error, k=5)
        train_errors.append(tr_e), validation_errors.append(val_e)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(0, 11)), y=train_errors, mode="markers", name="Train Error"))
    fig2.add_trace(go.Scatter(x=list(range(0, 11)), y=validation_errors, mode="markers", name="Validation Error"))
    fig2.update_layout(title="Cross-validation for polynomial fitting", xaxis={"title": "Degree", "tickmode": "linear"},
                       yaxis={"title": "MSE"})
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = int(np.argmin(validation_errors))
    poly_est = PolynomialFitting(best_k).fit(train_X, train_y)
    test_err = poly_est.loss(test_X, test_y)
    print(f"The best fitting polynomial degree is {best_k} and its test error is {test_err:.2f}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), n_samples / X.shape[0])
    train_X, train_y, test_X, test_y = np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_range = [0, 0.2]
    lam_arr = np.linspace(lam_range[0], lam_range[1], n_evaluations)
    ridge_error_df = pd.DataFrame(columns=["lam", "train_error", "validation_error"])
    lasso_error_df = pd.DataFrame(columns=["lam", "train_error", "validation_error"])

    for lam in lam_arr:
        ridge_est, lasso_est = RidgeRegression(lam), Lasso(alpha=lam)
        train_e, val_e = cross_validate(ridge_est, train_X, train_y, mean_square_error, 5)
        ridge_error_df = ridge_error_df.append({"lam": lam, "train_error": train_e, "validation_error": val_e},
                                               ignore_index=True)

        train_e, val_e = cross_validate(lasso_est, train_X, train_y, mean_square_error, 5)
        lasso_error_df = lasso_error_df.append({"lam": lam, "train_error": train_e, "validation_error": val_e},
                                               ignore_index=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=lam_arr, y=ridge_error_df["train_error"], mode="lines", name="Ridge Train Error"))
    fig3.add_trace(go.Scatter(x=lam_arr, y=ridge_error_df["validation_error"], mode="lines",
                              name="Ridge Validation Error"))
    fig3.add_trace(go.Scatter(x=lam_arr, y=lasso_error_df["train_error"], mode="lines", name="Lasso Train Error"))
    fig3.add_trace(go.Scatter(x=lam_arr, y=lasso_error_df["validation_error"], mode="lines",
                              name="Lasso Validation Error"))
    fig3.update_layout(title=f"Error vs. Regularization Parameter for Ridge and Lasso <br>"
                             f"{n_samples} samples, {n_evaluations} evaluations",
                       title_x=0.5,
                       xaxis={"title": "Regularization Parameter λ"},
                       yaxis={"title": "MSE"},
                       margin_t=100)
    fig3.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    # find lam value for best validation error
    ridge_best_lam = ridge_error_df.loc[ridge_error_df["validation_error"].idxmin()]["lam"]
    lasso_best_lam = lasso_error_df.loc[lasso_error_df["validation_error"].idxmin()]["lam"]
    # fit best models
    ridge_est = RidgeRegression(ridge_best_lam).fit(train_X, train_y)
    lasso_est = Lasso(alpha=lasso_best_lam)
    lasso_est.fit(train_X, train_y)
    least_squares_est = LinearRegression().fit(train_X, train_y)
    zero_lam_est = RidgeRegression(0).fit(train_X, train_y)
    # report test errors
    print(f"Ridge:\n"
          f"Best Lambda = {ridge_best_lam:.2f}\n"
          f"Test Error = {ridge_est.loss(test_X, test_y):.2f}\n")

    lasso_test_preds = lasso_est.predict(test_X)
    print(f"Lasso:\n"
          f"Best Lambda = {lasso_best_lam:.2f}\n"
          f"Test Error = {mean_square_error(test_y, lasso_test_preds):.2f}\n")

    print(f"Least Squares Test Error: {least_squares_est.loss(test_X, test_y):.2f}")



if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(n_samples=100, noise=5)
    # select_polynomial_degree(n_samples=100, noise=0)
    # select_polynomial_degree(n_samples=1500, noise=10)
    # select_regularization_parameter()
