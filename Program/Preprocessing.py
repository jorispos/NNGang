# Takes a time series and the frame width
def splitSeries(timeSeries, frameWidth):
    length = len(timeSeries)

    series = []

    leftIndex = 0
    rightIndex = frameWidth-1

    while rightIndex != length:
        series.append(timeSeries[leftIndex:rightIndex+1])
        leftIndex += 1
        rightIndex += 1

    return series

# I would create functions for detrending, deseasonalizing, and scaling here bellow