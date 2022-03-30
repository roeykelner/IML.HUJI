from __future__ import annotations

import numpy as np
from numpy.linalg import inv


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False):  # -> UnivariateGaussian
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    @staticmethod
    def gaussian_density(val, mu, var):
        return np.exp((-1) * ((val - mu) ** 2) / 2 * var) / np.sqrt((2 * np.pi * var))

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X)
        s = 0
        for x_i in X:
            s += (x_i - self.mu_) ** 2

        # Dividing by n-1 or n according to bias of estimator
        self.var_ = s / (len(X) - 1) if not self.biased_ else s / (len(X))

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        # Gaussian density function
        # return np.exp((-1) * ((X - self.mu_) ** 2) / 2 * self.var_) / np.sqrt((2 * np.pi * self.var_))
        return self.gaussian_density(X, self.mu_, self.var_)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # Likelihood of X under (mu, sigma) is defined as the density function (mu, sigma) of X
        exp = np.sum((X - mu) ** 2) / (-2 * sigma)
        return exp - (len(X) / 2) * np.log(2 * np.pi * sigma)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)  # mu_[j] = mean of the j'th column of X
        X_centered = X - np.tile(self.mu_, (X.shape[0], 1))  # removing the mean from every column
        self.cov_ = np.dot(X_centered.transpose(), X_centered) / (X.shape[0] - 1)  # formula for the covariance matrix

        self.fitted_ = True
        return self

    @staticmethod
    def mult_gaussian_density(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        """
        @param X: 1d Vector of dimension D
        @param mu: Gaussian expectation
        @param sigma: Gaussian variance
        @return: Density value of a single Gaussian Multivariate vector X
        """
        X_centered = X - mu
        expo = np.exp((-0.5) * X_centered @ np.linalg.inv(sigma) @ X_centered.transpose())
        d = X.shape[0]
        return expo / np.sqrt((2 * np.pi) ** d * np.linalg.det(sigma))

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        res = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            res[i] = self.mult_gaussian_density(X[i, :], self.mu_, self.cov_)

        return res

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """

        # Using the formula I calculated in q9:
        d = X.shape[1]  # number of features
        m = X.shape[0]  # number of samples
        sum = 0
        for x_i in X:
            sum += (x_i - mu).transpose() @ np.linalg.inv(cov) @ (x_i - mu)
        l = -0.5 * m * np.log(np.power(2 * np.pi, d) * np.linalg.det(cov))
        return l - 0.5 * sum
