# Split a time series into many small time series with a given width
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


def splitSeries(timeSeries, frameWidth, emptySpace):
    length = len(timeSeries)

    series = []

    leftIndex = 0
    rightIndex = frameWidth - 1

    while rightIndex != (length - emptySpace):
        series.append(timeSeries[leftIndex:rightIndex + 1])
        leftIndex += 1
        rightIndex += 1

    return series


# I would create functions for deseasonalizing and scaling here bellow
# They would probably take some time series vector/matrix and return the preprocessed version,
# You  may want to also save the model of the seasons/scale so that they can be added back later

# Gets all the trend points of a given timeseries
def getTrend(timeSeries):
    # Reformat X and y
    X = range(len(timeSeries))
    X = np.reshape(X, (len(X), 1))
    y = timeSeries

    # Compute quadratic model
    pf = PolynomialFeatures(degree=2)
    Xp = pf.fit_transform(X)
    md2 = LinearRegression()
    md2.fit(Xp, y)
    trend = md2.predict(Xp)

    # For debugging
    plt.plot(X, y)
    plt.plot(X, trend)
    plt.legend(['data', 'polynomial trend'])
    plt.show()

    return trend


# Removes the trend of a given time series and returns the detrended version
def removeTrend(timeSeries, trend):
    # Reformat X and y
    X = range(len(timeSeries))
    X = np.reshape(X, (len(X), 1))

    # Remove trend from original model
    detrendedSeries = [timeSeries[i] - trend[i] for i in range(0, len(timeSeries))]

    # For debugging
    plt.plot(X, detrendedSeries)
    plt.title('polynomially detrended data')
    plt.show()

    return detrendedSeries



def deseasonalize(df): #(dataframe):        dit is from https://www.kaggle.com/code/prashant111/complete-guide-on-time-series-analysis-in-python/notebook
    # Subtracting the Trend Component

    # Time Series Decomposition
    result_mul = seasonal_decompose(df, model='multiplicative', period=30)


    # Deseasonalize
    deseasonalized = df.values / result_mul.seasonal

    return deseasonalized