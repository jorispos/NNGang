class Splitter:
    def __init__(self):
        print("Splitter initialized")

    # Takes a time series and the frame width
    def splitSeries(self, timeSeries, frameWidth):
        length = len(timeSeries)