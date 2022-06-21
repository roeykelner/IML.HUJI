import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X = np.random.rand(n, 2) * 2 - 1
    y = np.ones(n, dtype=np.intp)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1

    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    ab_est = AdaBoost(wl=DecisionStump, iterations=n_learners).fit(train_X, train_y)
    train_errors = np.array([ab_est.partial_loss(train_X, train_y, T) for T in range(1, n_learners + 1)])
    test_errors = np.array([ab_est.partial_loss(test_X, test_y, T) for T in range(1, n_learners + 1)])

    fig_q1 = go.Figure()
    fig_q1.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=train_errors, mode='lines', name="Train"))
    fig_q1.add_trace(go.Scatter(x=list(range(1, n_learners + 1)), y=test_errors, mode='lines', name="Test"))
    fig_q1.update_layout(
        title={"text": f"<b>Test Errors as a Function of Ensemble Size"
                       f"<br>Noise = {noise}</b>",
               "x": 0.5,
               "xanchor": "center"})
    fig_q1.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(["circle", "x"])
    fig_q2 = make_subplots(rows=2, cols=2, subplot_titles=[f"{m} iterations" for m in T],
                           horizontal_spacing=0.01, vertical_spacing=.07)
    for i, iterations in enumerate(T):
        fig_q2.add_traces([decision_surface(ab_est.partial_predict, lims[0], lims[1], showscale=False, T=iterations),
                           go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                      marker=dict(color=test_y,
                                                  symbol=symbols[np.intp(np.abs(
                                                      ab_est.partial_predict(test_X, T=iterations) - test_y) / 2)],
                                                  colorscale=[custom[0], custom[-1]],
                                                  line=dict(color="black", width=1)))],
                          rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig_q2.update_layout(
        title={"text": f"<b>Decision Boundaries Of Models as a Function of Number of Ensemble Size "
                       f"<br>Noise = {noise}</b>",
               "x": 0.5,
               "xanchor": "center"},
        margin=dict(t=130)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig_q2.show()

    # Question 3: Decision surface of best performing ensemble
    min_loss_size = np.argmin(test_errors) + 1
    acc = accuracy(test_y, ab_est.partial_predict(test_X, min_loss_size))
    fig_q3 = go.Figure()
    fig_q3.add_traces([decision_surface(ab_est.partial_predict, lims[0], lims[1], showscale=False, T=min_loss_size),
                       go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                  marker=dict(color=test_y, symbol=symbols[np.intp(np.abs(
                                      ab_est.partial_predict(test_X, T=min_loss_size) - test_y) / 2)],
                                              colorscale=[custom[0], custom[-1]],
                                              line=dict(color="black", width=1)))])
    fig_q3.update_layout(
        title={
            "text": f"<b>Decision Boundary Of Best Preforming Model <br>Ensemble Size = {min_loss_size}"
                    f"    Noise = {noise}"
                    f" <br>Accuracy = {acc}</b>",
            "x": 0.5,
            "xanchor": "center"},
        margin=dict(t=130)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig_q3.show()

    # Question 4: Decision surface with weighted samples
    D = ab_est.D_ / np.max(ab_est.D_) * 14
    fig_q4 = go.Figure()
    fig_q4.add_traces([decision_surface(ab_est.partial_predict, lims[0], lims[1], showscale=False, T=n_learners),
                       go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                  marker=dict(color=train_y, symbol=symbols[np.intp(np.abs(
                                      ab_est.partial_predict(train_X, T=n_learners) - train_y) / 2)],
                                              colorscale=[custom[0], custom[-1]], size=D,
                                              line=dict(color="black", width=1)))])
    fig_q4.update_layout(
        title={
            "text": f"<b>Ensemble size = {n_learners}<br>"
                    f"Marker size proportional to final weight <br>Noise = {noise}</b>",
            "x": 0.5,
            "xanchor": "center"},
        margin=dict(t=130)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig_q4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0, train_size=5000)
    fit_and_evaluate_adaboost(noise=0.4, train_size=5000)
