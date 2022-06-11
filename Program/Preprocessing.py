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