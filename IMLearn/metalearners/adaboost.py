import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn
from ..metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.weights_: np.ndarray
        self.models_, self.weights_, self.D_ = None, None, None

    def _calculate_weight_error(self, y_true, y_pred):
        sum = 0
        for y_pred_i, y_true_i, D_i in zip(y_pred, y_true, self.D_):
            if np.sign(y_pred_i) != np.sign(y_true_i):
                sum += D_i
        return sum

    def _update_D(self, y: np.ndarray, y_pred: np.ndarray, w_t):
        self.D_ = self.D_ * np.exp(-1 * y * w_t * y_pred)
        self.D_ = self.D_ / np.sum(self.D_)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        m_samples, d_features = X.shape
        self.models_ = []
        self.weights_ = np.zeros((self.iterations_,))
        self.D_ = np.ones((m_samples,)) / m_samples
        for t in range(self.iterations_):
            print(f"fitting... {(t/self.iterations_*100):.1f}%")
            h_t = self.wl_().fit(X, y * self.D_)
            self.models_.append(h_t)
            pred_t = h_t.predict(X)
            e_t = self._calculate_weight_error(y, pred_t)
            self.weights_[t] = 0.5 * np.log(1 / e_t - 1)
            self._update_D(y, pred_t, self.weights_[t])
        print(f"fitting... 100% !!")

    def _predict(self, X):
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

        # preds_mat[t,i] is the prediction of h_t on x_i
        w_preds_mat = np.array([h_t.predict(X) * w_t for h_t, w_t in
                                zip(self.models_[:self.iterations_], self.weights_[:self.iterations_])])
        return np.sign(np.sum(w_preds_mat, axis=0))

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

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        self.iterations_ = T
        return self.predict(X)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under misclassification loss function
        """
        self.iterations_ = T
        return misclassification_error(y, self.predict(X))
