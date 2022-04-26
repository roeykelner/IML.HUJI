from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features' covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features' covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        d_features = X.shape[1]
        m_samples = X.shape[0]
        n = np.zeros(n_classes)
        for k in y:
            n[k] += 1  # Counting occurrences of each class/label
        self.pi_ = n / m_samples
        feature_sum = np.zeros(shape=(n_classes, d_features))
        for x_i, y_i in zip(X, y):
            feature_sum[y_i] += x_i  # Summing the feature vectors of each class
        self.mu_ = feature_sum / n[:, None]  # Dividing every feature by number of occurrences
        sum_cov_matrices = np.zeros(shape=(d_features, d_features))
        for x_i, y_i in zip(X, y):
            sum_cov_matrices += np.outer(x_i - self.mu_[y_i], x_i - self.mu_[y_i])
        self.cov_ = sum_cov_matrices / (m_samples - n_classes)
        self._cov_inv = inv(self.cov_)

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
        return np.argmax(self.likelihood(X), axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        m_samples = X.shape[0]
        n_classes = self.classes_.shape[0]
        lkl = np.zeros(shape=(m_samples, n_classes))
        for i, x_i in enumerate(X):
            for k in self.classes_:
                a_k = self._cov_inv @ self.mu_[k]
                b_k = np.log(self.pi_[k]) - 0.5 * self.mu_[k].T @ self._cov_inv @ self.mu_[k]
                lkl[i, k] = a_k.T @ x_i + b_k
        return lkl

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
        from ...metrics import misclassification_error
        return misclassification_error(y_true=y, y_pred=self.predict(X))
