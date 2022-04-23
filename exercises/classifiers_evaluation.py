import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        samp, resp = load_dataset(f'../datasets/{f}')

        # Fit Perceptron and record loss in each fit iteration
        def callback_func(perc_obj: Perceptron, sample: np.ndarray, response: int):
            # callback_func.iter_num += 1  # Counting the number of calls to the function
            losses.append(perc_obj.loss(samp, resp))

        # callback_func.iter_num = 0
        losses = []
        perc = Perceptron(callback=callback_func)
        perc.fit(X=samp, y=resp)

        # Plot figure of loss as function of fitting iteration
        fig = px.scatter(x=range(len(losses)), y=losses, title=f"<b>Loss over number of iterations</b><br>{n} Dataset",
                         labels={'x': "Fit Iterations", 'y': "Loss"})
        fig.update_layout(margin_t=120, title_x = 0.5)
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X_train, y_train = load_dataset(f'../datasets/{f}')

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X_train, y_train)
        lda_pred = lda.predict(X_train)

        naive_gauss = GaussianNaiveBayes()
        naive_gauss.fit(X_train, y_train)
        naive_gauss_pred = naive_gauss.predict(X_train)

        from IMLearn.metrics import accuracy

        est_dic = {naive_gauss: {"prediction": naive_gauss_pred,
                                 "accuracy": accuracy(y_true=y_train, y_pred=naive_gauss_pred)},
                   lda: {"prediction": lda_pred,
                         "accuracy": accuracy(y_true=y_train, y_pred=lda_pred)}}

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[
                                f"Naive Bayes Gaussian Classifier <br>Accuracy = {est_dic[naive_gauss]['accuracy']:.3f}",
                                f"LDA Classifier <br>Accuracy = {est_dic[lda]['accuracy']:.3f}"])

        for col, estimator in enumerate([naive_gauss, lda], start=1):
            # Add traces for data-points setting symbols and colors
            fig.add_trace(go.Scatter(x=X_train[:, 0], y=X_train[:, 1], mode="markers", showlegend=False,
                                     marker=dict(color=est_dic[estimator]["prediction"], symbol=y_train,
                                                 line=dict(color="black", width=1))), row=1, col=col)

            # Add `X` dots specifying fitted Gaussians' means
            fig.add_trace(go.Scatter(x=estimator.mu_[:, 0], y=estimator.mu_[:, 1], mode="markers", showlegend=False,
                                     marker=dict(color='black', symbol='x', size=15, opacity=0.8,
                                                 line=dict(color="black", width=1))), row=1, col=col)

            # Add ellipses depicting the covariances of the fitted Gaussians
            for k in estimator.classes_:
                cov = estimator.cov_ if estimator is lda else np.diag(estimator.vars_[k])
                fig.add_trace(get_ellipse(estimator.mu_[k], cov), row=1, col=col)

        fig.update_layout(title_text=f"<b>Classifier Comparison </b><br>Dataset: {f}", title_x=0.5, title_y=0.9,
                          margin_t=150)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
