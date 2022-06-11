# Split a time series into many small time series with a given width
def splitSeries(timeSeries, frameWidth, emptySpace):
    length = len(timeSeries)

    series = []

    leftIndex = 0
    rightIndex = frameWidth-1

    while rightIndex != (length - emptySpace):
        series.append(timeSeries[leftIndex:rightIndex+1])
        leftIndex += 1
        rightIndex += 1

    return series

# I would create functions for detrending, deseasonalizing, and scaling here bellow
# They would probably take some matrix and return the preprocessed matrix,
# You  may want to also save the model of the trend/seasons/scale so that they can be added back later