from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error


def _weighted_mce(y_true: np.ndarray, y_pred: np.ndarray, m_samples: int):
    return np.sum(np.abs(y_true[y_true * y_pred < 0])) / m_samples


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is above the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_thr, min_err, min_j, min_sign = None, np.infty, None, None
        for j, j_col in enumerate(X.T):
            for sign in [-1, 1]:
                j_thr, j_err = self._find_threshold(j_col, y, sign)
                if j_err < min_err:
                    min_thr, min_err, min_j, min_sign = j_thr, j_err, j, sign

        self.threshold_, self.j_, self.sign_ = min_thr, min_j, min_sign

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

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] >= self.threshold_, self.sign_, -1 * self.sign_)

    @staticmethod
    def _find_threshold(values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to the values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassification error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        best_thr, min_err = None, np.infty
        for v in values:
            prediction = np.where(values < v, -1 * sign, sign)
            error = _weighted_mce(labels, prediction, len(values))
            if error < min_err:
                best_thr, min_err = v, error

        return best_thr, min_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification loss function
        """
        return misclassification_error(y, self.predict(X))
