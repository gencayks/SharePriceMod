import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import prophet as fbp

from sklearn.metrics import mean_squared_error, mean_absolute_error

#use fivethirtyeight plot style

plt.style.use('fivethirtyeight')

df = pd.read_csv('sample_data / amzn.csv')

df.head()

# add two columnsin dataframe having values as Date and Adj Close
df[['ds', 'y']] = df[['Date', 'Adj Close']]
# Subset two columns from data frame
df = df[['ds', 'y']]

df.head()

# split data frame  into two parts train and test
split_date = "2019-07-21"
df_train = df.loc[df.ds <= split_date].copy()
df_test = df.loc[df.ds > split_date].copy()

# Instantiate prophet
model = fbp.Prophet()
# fit the training data
model.fit(df_train)


forecast = model.predict(df_test)
forecast.tail()


model.plot(forecast)

# plot graphs of different components:
model.plot_components(forecast)


# code
print("Mean Squared Error (MSE):", mean_squared_error(y_true = df_test["y"], y_pred = forecast['yhat']))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_true = df_test["y"], y_pred = forecast['yhat']))


def mean_abs_perc_err(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


print("Mean Absolute % Error (MAPE): ",
      mean_abs_perc_err(y_true=np.asarray(df_test["y"]), y_pred=np.asarray(forecast['yhat'])))



