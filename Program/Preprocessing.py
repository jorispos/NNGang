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

    # Compute linear model
    pf1 = PolynomialFeatures(degree=1)
    Xp1 = pf1.fit_transform(X)
    md1 = LinearRegression()
    md1.fit(Xp1, y)
    trend1 = md1.predict(Xp1)

    # Compute quadratic model
    pf2 = PolynomialFeatures(degree=2)
    Xp2 = pf2.fit_transform(X)
    md2 = LinearRegression()
    md2.fit(Xp2, y)
    trend2 = md2.predict(Xp2)

    trend = [(trend1[i] + trend2[i])/2 for i in range(0, len(trend1))]

    return trend


# Removes the trend of a given time series and returns the detrended version
def removeTrend(timeSeries, trend):
    # Reformat X and y
    X = range(len(timeSeries))
    X = np.reshape(X, (len(X), 1))

    # Remove trend from original model
    detrendedSeries = [timeSeries[i] - trend[i] for i in range(0, len(timeSeries))]

    return detrendedSeries

# Removes the trend of a given time series and returns the detrended version
def addArray(timeSeries, trend):
    # Add trend from preprocessd model
    addedSeries = [timeSeries[i] + trend[i] for i in range(0, len(timeSeries))]

    return addedSeries

def getSeasons(timeSeries):
    # Time Series Decomposition
    result_mul = seasonal_decompose(timeSeries, model='additive', period=7)
    return result_mul.seasonal


def removeSeasons(timeSeries, seasons):
    # Deseasonalize
    deseasonalized = [timeSeries[i] - seasons[i] for i in range(0, len(timeSeries))]

    return deseasonalized