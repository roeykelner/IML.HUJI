from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable, List
import numpy as np
from numpy import ceil

from IMLearn import BaseEstimator

def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], k: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    k: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # Splitting data to folds
    folds_X, folds_Y = np.array_split(X, k), np.array_split(y, k)
    train_errors, validation_errors = np.zeros(k), np.zeros(k)
    for i in range(k):
        # Fitting on S / S_i
        sub_x, sub_y = np.vstack(folds_X[:i] + folds_X[i + 1:]), np.concatenate(folds_Y[:i] + folds_Y[i + 1:])
        estimator.fit(sub_x, sub_y)
        train_errors[i] = scoring(sub_y, estimator.predict(sub_x))
        validation_errors[i] = scoring(folds_Y[i], estimator.predict(folds_X[i]))
    return np.average(train_errors), np.average(validation_errors)


