from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from scipy.stats import norm


# noinspection DuplicatedCode
class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        d_features = 1
        if X.ndim ==2:
            d_features = X.shape[1]
    

        m_samples = X.shape[0]
        n = np.zeros(n_classes).astype(int)
        for k in y:
            n[k] += 1  # Counting occurrences of each class/label
        self.pi_ = n / m_samples
        feature_sum = np.zeros(shape=(n_classes, d_features))
        for x_i, y_i in zip(X, y):
            feature_sum[y_i] += x_i  # Summing the feature vectors of each class
        self.mu_ = feature_sum / n[:, None]  # Dividing every feature by number of occurrences
        self.vars_ = np.zeros(shape=(n_classes, d_features))
        for k in self.classes_:
            x_k = np.zeros(shape=(n[k], d_features))
            fill_ind = 0
            for X_i, y_i in zip(X, y):
                if y_i == k:
                    x_k[fill_ind] = X_i
                    fill_ind += 1
            self.vars_[k] = np.var(x_k, axis=0, ddof=1)

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

        # Using the expression derived in question 3b:
        m_samples = X.shape[0]
        d_features = X.shape[1] if X.ndim == 2 else 1
        n_classes = self.classes_.shape[0]
        lkl = np.zeros((m_samples, n_classes))
        for i, x_i in enumerate(X):
            for k in self.classes_:
                lkl[i, k] = np.log(self.pi_[k])
                for j in range(d_features):
                    lkl[i, k] += np.log(1 / (np.sqrt(self.vars_[k, j] * 2 * np.pi))) \
                                 - 0.5 * ((x_i[j] - self.mu_[k, j]) / np.sqrt(self.vars_[k, j])) ** 2
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
        return misclassification_error(y, self.predict(X))
