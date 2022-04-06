from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
import os
from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # todo remove
    pd.set_option("max_columns", None)  # show all cols
    pd.set_option('max_colwidth', None)  # show full width of showing cols
    pd.set_option("expand_frame_repr", False)  # print cols side by side as it's supposed to be

    df = pd.read_csv(filename)

    # Removing samples with invalid values
    df = df[(df['price'] > 0) &
            (df['sqft_lot15'] > 0) &
            (df['yr_built'] > 1500) &
            (df['bedrooms'] < 20)
            ]

    # Replacing categorical feature with dummy features.
    df = pd.get_dummies(df, columns=["zipcode"])

    # Removing ID and 'date' columns
    df = df.drop(columns=['id', 'date'])
    prices = df['price']

    df = df.drop(columns=['price'])

    return tuple([df, prices])


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for feature in X:
        corr = (np.cov(X[feature], y) / (np.std(X[feature]) * np.std(y)))[0, 1]
        fig = px.scatter(x=X[feature], y=y,
                         title=f"House price as a function of {feature.capitalize()}\n"
                               f"Correlation = {corr:.3f}",
                         labels={'x': f'{feature.capitalize()}', 'y': 'House price'})
        fig.update_layout(title_x=0.5)
        # fig1.show()
        fig.write_image(f"{output_path}/{feature}_vs_price.jpeg")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X,y, output_path="./images")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    # print(X, y, sep="\n")
    # print(train_X, train_y, test_X, test_y, sep="\n\n")

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    res = np.empty([len(range(10, 101)), 3])
    for i, p in enumerate(range(10, 101)):
        losses = []
        for j in range(10):
            frac_X = train_X.sample(frac=p / 100)
            frac_y = train_y[frac_X.index]
            estimator = LinearRegression()
            estimator.fit(frac_X, frac_y)
            losses.append(estimator.loss(test_X, test_y))
        res[i, 0] = p
        res[i, 1] = np.average(losses)
        res[i, 2] = np.var(losses)
    # print(res)
    percent_plot = px.scatter(x=res[:, 0], y=res[:, 1],
                              title=f"MSE as a function of percentage sampled",
                              labels={'x': 'Percentage', 'y': 'MSE'})
    percent_plot.update_layout(title_x=0.5)
    # percent_plot.show()
    var_plot = px.scatter(x=res[:, 0], y=res[:, 2],
                              title=f"Variance as a function of percentage sampled",
                              labels={'x': 'Percentage', 'y': 'Variance'},
                          log_y=True)
    var_plot.update_layout(title_x=0.5)
    var_plot.show()
