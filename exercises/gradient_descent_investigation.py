import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import sklearn.metrics

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm
    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted
    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path
    title: str, default=""
        Setting details to add to plot title
    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range
    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range
    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown
    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}", margin=dict(l=0, r=0, b=0, t=60)))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weightses = []

    def f(val, weights, **kwargs):
        values.append(val)
        weightses.append(weights)

    return f, values, weightses


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    convergence_fig = go.Figure()
    for module_type in [L1, L2]:
        for fixed_eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(FixedLR(fixed_eta), callback=callback)
            module = module_type(init.copy())
            gd.fit(f=module, X=None, y=None)
            plot_descent_path(module=module_type, descent_path=np.array(weights),
                              title=f"<br>Module = {module_type.__name__}, eta = {fixed_eta}").show()

            convergence_fig.add_trace(go.Scatter(x=list(range(len(values))), y=values, mode="lines + markers",
                                                 name=f"Module {module_type.__name__}, eta = {fixed_eta}"))

            print(f"Module {module_type.__name__}, eta = {fixed_eta}, lowest loss = {min(values):.3e}")
    convergence_fig.update_xaxes(type="log")
    convergence_fig.update_layout(title=f"GD Objective Value vs. Iteration for both modules", )
    convergence_fig.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig5 = go.Figure()
    weights_95 = []
    min_val = np.inf
    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(ExponentialLR(eta, gamma), callback=callback)
        module = L1(init.copy())
        gd.fit(f=module, X=None, y=None)
        if gamma == .95:
            weights_95 = weights
        if min(values) < min_val:
            min_val = min(values)
        fig5.add_trace(go.Scatter(x=list(range(1000)), y=values, mode="lines + markers",
                                  name=f"gamma = {gamma}"))

    print(f"The lowest norm achieved with an exponential LR is {min_val:.3e}")

    # Plot algorithm's convergence for the different values of gamma
    fig5.update_layout(title="GD Objective Value vs. Iteration for Module L1",
                       xaxis=dict(title="Iteration"), yaxis=dict(title="Objective Value"))
    # make both axes logarithmic
    fig5.update_xaxes(type="log")
    fig5.update_yaxes(type="log")
    fig5.show()
    # Plot descent path for gamma=0.95
    fig7 = plot_descent_path(module=L1, descent_path=np.array(weights_95), title="gamma = 0.95")
    fig7.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Plotting convergence rate of logistic regression over SA heart disease data
    from sklearn.metrics import roc_curve, auc
    from utils import custom
    c = [custom[0], custom[-1]]

    est = LogisticRegression()
    est.fit(X_train, y_train)
    probs = est.predict_proba(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, probs)
    roc_fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3e}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    # add a scatter showing (tpr - fpr) for each threshold
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr - fpr, mode="lines", name="", text=thresholds,
                                 hovertemplate="<b>Threshold:</b>%{text:.3e}",
                                 line=dict(color=c[0][1])))

    roc_fig.show()

    best_alpha = thresholds[np.argmax(tpr - fpr)]
    print(f"Best alpha = {best_alpha:.3e}")
    best_alpha_logis_reg = LogisticRegression(alpha=best_alpha)
    best_alpha_logis_reg.fit(X_train, y_train)
    print(f"Best alpha, no regularisation test error: {best_alpha_logis_reg.loss(X_test, y_test):.3f}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    from IMLearn.model_selection import cross_validate
    from IMLearn.metrics.loss_functions import misclassification_error

    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for module in [L1, L2]:
        # print(f"Fitting {module.__name__} model...")
        best_lambda, lowest_error = None, np.inf
        # print(f"Fitting {len(lambdas)} λ-models with cross-validation...")
        for i, lam in enumerate(lambdas):
            # print(f"{i / len(lambdas) * 100:.2f}% . . .  λ = {lam:.2e}  ", end="")

            train_err, val_err = cross_validate(LogisticRegression(penalty=module.__name__.lower(), alpha=.5, lam=lam,
                                                                   solver=GradientDescent(max_iter=20000,
                                                                                          learning_rate=FixedLR(1e-4))),
                                                X_train,
                                                y_train,
                                                scoring=misclassification_error)
            if val_err < lowest_error:
                lowest_error = val_err
                best_lambda = lam
            # print(f"err = {val_err:.3f}")

        # print("100% . . . Done!")
        print(f"({module.__name__}) Best lambda = {best_lambda:.3e}")
        best_lambda_logis_reg = LogisticRegression(penalty=module.__name__.lower(), alpha=.5, lam=best_lambda)
        best_lambda_logis_reg.fit(X_train, y_train)
        print(f"({module.__name__}) Best lambda test error: {best_lambda_logis_reg.loss(X_test, y_test):.3f}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
