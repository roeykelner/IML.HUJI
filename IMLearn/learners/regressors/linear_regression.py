from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import pinv
from ...metrics.loss_functions import mean_square_error


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True):
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should the fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should the fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    @staticmethod
    def _add_ones(X: np.ndarray) -> np.ndarray:
        """
        Adds a column of ones on the left of the X matrix
        @param X:
        @return: X with ones columns to the left
        """
        return np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            # Adding a 1's column at the beginning of the X sample matrix
            X = self._add_ones(X)
        # As we've seen in class, the w_ estimator for RSS-loss function is the pseudo-inverse of X times y.
        self.coefs_ = pinv(X) @ y

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            # Adding a 1's column at the beginning of the X sample matrix
            X = self._add_ones(X)
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        y_pred = self.predict(X)
        return mean_square_error(y_pred, y)
