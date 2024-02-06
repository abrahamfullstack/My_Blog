---
title: "Forecasting Model with Exponential Smoothing"
datePublished: Tue Feb 06 2024 22:20:34 GMT+0000 (Coordinated Universal Time)
cuid: clsaxcqmj000809l7e3jd7sea
slug: forecasting-exponential-smoothing
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1707240255183/4741194a-7657-4704-8204-5d89d9def5bc.png
tags: python, ml, forecast

---

This is the first article of a series to present different forecasting models. We will start by using the **Exponential Smoothing** model provided by **statsmodels** library.

The dataset is the **Alphabet Inc.** daily stock prices that can be downloaded from [Yahoo Finance](https://finance.yahoo.com/quote/GOOG/history?p=GOOG). Data ranged from Jan-2020 to Dec-2023.

The target column is `Close` column. It represents the stock price at the end of the day.

Here are the steps we will follow in each article:

1. Data Preparation
    
2. Decomposing The Data Set
    
3. Training The Model
    
4. Evaluate The Model
    

# Exponential Smoothing

**Exponential Smoothing** is a time series forecast method for univariate data. It means that can make forecast predictions for just one variable related to the time series.

One of the advantages of using the **Exponential Smoothing** is that uses Holt’s Linear Trend Model to learn the trend and seasonality of the data. This is an popular alternative to **ARIMA** model.

# Data Preparation

Make sure to download the csv file from [Yahoo Finance](https://finance.yahoo.com/quote/GOOG/history?p=GOOG) and save it in the same directory as your Jupyter notebook.

Lets import all of the needed modules and packages

```python
# Import usefull libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
```

We read the dataset using `pandas`.

```python
# Collect the dataset
df = pd.read_csv('GOOG.csv')
df.head()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1707240376779/d9404b33-7af9-4292-b0cb-0a45271fdd80.png align="center")

We can observe that the dataset contains more than one column. For further models we will use them for now lets just focus on the `Date` and `Close` column.

In case there are nulls on our dataset we replace them, but the downloaded file should be all clean data.

```python
# Check for nulls
df.info()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1707240577580/a83c26cb-5890-4988-8020-dc22d3315a09.png align="center")

Now that we are sure that all records contains non-null values we can use the Date column as our index.

```python
# Use date column as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Remove duplicates based on index, keep first found
df = df[~df.index.duplicated(keep='first')]
df.asfreq('D')
df.sort_index(inplace=True)

df.head()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1707240671160/2a8d3d1f-29f1-4b6b-81ae-57bb5ebaae5f.png align="center")

# Decomposing The Data Set

On this section we will reveal the **trend**, **seasonality** and **residuals**. Here is a description of each element:

* Trend - represents the **increasing or decreasing pattern** over time
    
* Seasonality - represents patterns that occur at regular intervals (e.g., daily, weekly, monthly, etc.)
    
* Residuals - represents the noise or the random variation in the data
    

We can start by plotting the data so we can visually discover the trend and seasonality. We can include some vertical lines to identify each year on the data.

```python
# Plot the data
vlines = ['2021-01-01', '2022-01-01', '2023-01-01']

df.plot(y='Close', figsize=(15,5))

for line in vlines:
  plt.axvline(x=line, color='black', linestyle='--')

plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1707241033294/8ea5000e-c93b-49a6-8b02-5c60f4ce2041.png align="center")

With above plot we can see that expect for 2022, each start of the year the stock prices tend to go up. Also, on and after 2022 there seems to be more variation in between days, in other word we have more noise during this period.

It is required to identify the type of seasonality. There are two main types:

* **Additive** \- assumes that changes over time are consistently made by the same amount.
    
* **Multiplicative** \- suggests that the components interact in a multiplicative way. Changes in trend and seasonality are proportional to the level of the time series.
    

I highly recommend readying this [article](https://towardsdatascience.com/finding-seasonal-trends-in-time-series-data-with-python-ce10c37aa861) to learn more.

On this article we will use the `seasonal_decompose` module from `statsmodels` to plot each of the component of the time series. It is required to pass the type of seasonality and by the previous plot there is a clear tendency of **add** to the next values.

```python
# Plot the decompose item
series = df[['Close']].copy()
decompose_result_mult = seasonal_decompose(series, model="additive", period=365)


decompose_result_mult.plot()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1707241676001/1a9e82ad-3e56-48b7-acdc-e2bbdc665e4d.png align="center")

`decompose_result_mult.plot()` includes the actual data plot, the trend line figure, the seasonal discomposing plot and the residuals plot.

Each of the plot helps us identifying the component of the time series that will be useful for further analysis. Lets train a model.

# Training The Model

Starting from defining the future horizon. This contains the future dates we want to predict once the model is trained.

```python
future = pd.date_range('2024-01-01', '2025-01-01', freq='D')
```

Now we define the **Exponential Smoothing** model by calling the `ExponentialSmoothing()` class

```python
# Define the Exponential Smoothing model
model = ExponentialSmoothing(df['Close'], 
                                seasonal_periods=365,
                                trend='add',
                                seasonal='add').fit(optimized=1)
```

We can get the fitted values from the model with `mode.fittedvalues`. This will return the values the model predict on the training data. Lets create a column to store those values an compare.

```python
# Fit the model
df['fitted_values'] = model.fittedvalues

df.head()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1707242117851/b2405a89-2833-401f-899a-4897897c96c2.png align="center")

If we plot both values we can compare how good the model is learning from the training data.

```python
# Compare fitted values and actual values

df['fitted_values'].plot(style='o', color='red')
df['Close'].plot(figsize=(10,5), color='blue')
plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1707242258451/a9a58465-09ca-40e5-8a5f-006dcca04999.png align="center")

# Evaluate The Model

There are many ways to evaluate a model, using metrics but we start with the basic on this notebook, so lets keep it simple with visual inspection. In future article we will improve the evaluation techniques.

First predict for the forecast horizon which is 180 days.

```python
# Forecast 180 days
pred = model.forecast(180)
pred
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1707242397545/8bf173b2-94bf-4e70-832a-77c38dccf83c.png align="center")

Pandas data frames are really useful to visualize the predictions, so lets create a data frame.

```python
# Concat forecast and fitted values
future = pd.date_range(df.index.max(), df.index.max() + pd.DateOffset(days=179), freq='D')

forecast = pd.DataFrame({'Date': future, 'pred': pred})

# Use date column as index
forecast['Date'] = pd.to_datetime(forecast['Date'])
forecast.set_index('Date', inplace=True)

forecast
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1707257478700/baf20593-036b-490f-8fd1-93833fb5eae5.png align="center")

Now we plot the predictions

```python
# Plot prediction values

plt.figure(figsize=(20,5))
plt.title('Forecast vs Actuals')
plt.plot(forecast['pred'], '--o', color='red')

plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1707257522883/6b869816-aab8-473a-8706-934ae1da4eea.png align="center")

If we visually inspect the predicted forecast we can see that the model did it best to understand the trend of the time series.

```python
# Plot both values

plt.figure(figsize=(20,5))
plt.title('Forecast vs Actuals')
plt.plot(forecast['pred'], '--o', color='red')
plt.plot(df['Close'], '-o')

plt.show()
```

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1707257595878/34e18405-871e-47e9-a00f-7bc07a294062.png align="center")

# Next Steps

**Exponential Smoothing** is really good when it comes to univariate time series forecasts as it learns from the trend, seasonality and residuals. This firsts article did not focused on statistical methods to validate the output of the forecast. We will explore more in the next one.

Meanwhile, I highly recommend investigating:

* Measures to evaluate forecasting models
    
* What other models can be used for multivariate datasets?
    

[Here is the full notebook for the code](https://github.com/abrahamfullstack/Machine_Learning/blob/main/FORECAST_Exponential_Smoothing.ipynb).

Enjoy training !!