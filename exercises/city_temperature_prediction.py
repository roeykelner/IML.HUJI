import pandas

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


# pd.set_option("max_columns", None)  # show all cols
# pd.set_option('max_colwidth', None)  # show full width of showing cols
# pd.set_option("expand_frame_repr", False)  # print cols side by side as it's supposed to be


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    df = pd.read_csv(filename, parse_dates=['Date'])

    # Adding the day of year column
    df['DayOfYear'] = df['Date'].dt.day_of_year

    # Removing samples with invalid values
    df = df[(df['Temp'] > -30)]
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset

    # Question 2 - Exploring data for specific country
    all_df = load_data("../datasets/City_Temperature.csv")
    il_df = all_df[all_df['Country'] == "Israel"]
    il_df["Year"] = il_df["Year"].astype(str)
    fig1 = px.scatter(x=il_df['DayOfYear'], y=il_df['Temp'], color=il_df['Year'],
                      labels={'x': "Day of the year", 'y': 'Temperature (Cel.)', 'color': 'Year'})
    fig1.show()

    # Grouping by month and temperature
    months_std_df = il_df[['Month', 'Temp']].groupby('Month').agg('std').reset_index()

    fig2 = px.bar(x=months_std_df['Month'], y=months_std_df['Temp'], labels={'x': "Month", 'y': "Temperature (Cel.)"},
                  title="STD of temperature in Israel month")
    # Making the X axis categorical rather than continuous
    fig2.update_xaxes(type='category')
    fig2.show()

    # Question 3 - Exploring differences between countries

    country_df = all_df[['Country', 'Month', 'Temp']].groupby(['Country', 'Month']).agg(['mean', 'std'])
    country_df = country_df.reset_index()

    fig3 = px.line(x=country_df[('Month',)], y=country_df[('Temp', 'mean')], color=country_df[('Country',)],
                   error_y=country_df[('Temp', 'std')],
                   labels={'x': "Month", 'y': "Temperature (Cel.)", 'color': 'Country'}
                   , title="Monthly mean temperature, seperated by country")
    fig3.update_xaxes(type='category')
    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    y = il_df['Temp']
    X = il_df['DayOfYear']
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    for item in [train_X, train_y, test_X, test_y]:
        item = np.array(item)

    losses = []
    for k in range(1, 11):
        poly_est = PolynomialFitting(k)
        poly_est.fit(train_X, train_y)
        losses.append(round(poly_est.loss(test_X, test_y), 2))
    print(f"Losses as a function of K: \n{losses}")

    fig4 = px.bar(x=range(1, 11), y=losses,
                  labels={'x': "Polynomial degree fitted", 'y': "MSE"},
                  title="MSE as a function of the polynomial degree used for fitting")
    fig4.update_xaxes(type='category')
    fig4.show()

    # # Question 5 - Evaluating fitted model on different countries
    il_model = PolynomialFitting(6)
    il_model.fit(X, y)

    c_loss = {'Jordan': None, 'The Netherlands': None, 'South Africa': None}
    for country in c_loss:
        c_df = all_df[all_df['Country'] == country]
        c_loss[country] = round(il_model.loss(X=c_df['DayOfYear'],
                                              y=c_df['Temp']), 2)
    fig5 = px.bar(x=c_loss.keys(), y=c_loss.values(),
                  labels={'x': "", 'y': "MSE"},
                  title="Model fitted on Israel, MSE by country"
                  )
    fig5.show()
