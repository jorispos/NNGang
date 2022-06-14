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


def getSeasons(timeSeries):
    # Time Series Decomposition
    result_mul = seasonal_decompose(timeSeries, model='additive', period=7)
    return result_mul.seasonal


def removeSeasons(timeSeries, seasons):
    # Deseasonalize
    deseasonalized = timeSeries - seasons

    # For debugging
    plt.plot(timeSeries)
    plt.title('Original frame', fontsize=16)
    plt.plot()
    plt.show()
    plt.plot(seasons)
    plt.title('Seasons', fontsize=16)
    plt.plot()
    plt.show()
    plt.plot(deseasonalized)
    plt.title('Deseasonalized frame', fontsize=16)
    plt.plot()
    plt.show()

    return deseasonalized