import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import Program.Preprocessing as preprocessing


# Read CSV file and return the integer matrix
import numpy


def getRows(path):
    # Load the dataset
    file = open(path)
    type(file)

    # Extract and store data from dataset
    csvreader = csv.reader(file)
    rows = []
    for row in csvreader:
        intRow = [int(x) for x in row]
        rows.append(intRow)

    return rows


# Print an array as a table
def printArray(array):
    [print(*line) for line in array]


# Visually display the graphs
def displayGraphs(x, y, predictions, timeLength):
    numPlots = len(x)

    for i in range(numPlots):
        plt.plot(range(1, timeLength, 1), x[i], 'yellow')
        plt.plot(timeLength, y[i], 'yo')
        plt.plot(timeLength, predictions[i], 'ro')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title('Henk in action')
        redLegend = mpatches.Patch(color='red', label='Predicted data')
        yellowLegend = mpatches.Patch(color='#D5CF0C', label='Actual data')
        plt.legend(handles=[redLegend, yellowLegend])
        plt.show()


def createMockDataFile(filePath, startRange, endRange, fluctuationStart, fluctuationEnd, numRows, numCols):
    with open(filePath, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

    data = []

    for col in range(numRows):
        positiveTrend = random.randint(0, 1)
        if positiveTrend == 0:
            positiveTrend = -1

        randomStart = random.randint(startRange, endRange)
        rows = []
        for row in range(numCols):
            rows.append(randomStart)
            randomStart += (random.randint(fluctuationStart, fluctuationEnd)) * positiveTrend
        data.append(rows)

    writer.writerows(data)

    print("New .csv file successfully created at: " + filePath + " :)")


def createMockData(startRange, endRange, fluctuationStart, fluctuationEnd, numRows, numCols):
    data = []

    for col in range(numRows):
        positiveTrend = random.randint(0, 1)
        if positiveTrend == 0:
            positiveTrend = -1

        randomStart = random.randint(startRange, endRange)
        rows = []
        for row in range(numCols):
            rows.append(randomStart)
            randomStart += (random.randint(fluctuationStart, fluctuationEnd)) * positiveTrend
        data.append(rows)


def plotDataMatrix(matrix, title, fontSize):
    for i in range(len(matrix)):
        print("plotting scaled (" + str(i + 1) + "/" + str(len(matrix)) + ")..")
        plotData(matrix[i], title + str(i), fontSize)


def plotData(yVals, title, fontSize):
    plt.plot(yVals)
    plt.title(title, fontsize=fontSize)
    plt.show()


def mergeArrays(array1, array2):
    newArray = []
    for row in array1:
        newArray.append(row)
    for row in array2:
        newArray.append(row)
    return newArray


# add points to end of each array 1 and shift array1 over to the right (assumes equal height)
def addAndShift(array1, points):
    newArray1 = []
    for i in range(len(array1)):
        row = array1[i]
        row = numpy.append(row, points[i])
        row = numpy.delete(row, 0)
        newArray1.append(row)
    return newArray1

def transpose(matrix):
    rows = len(matrix)
    columns = len(matrix[0])

    matrix_T = []
    for j in range(columns):
        row = []
        for i in range(rows):
            row.append(matrix[i][j])
        matrix_T.append(row)

    return matrix_T

def detrendAndDeseasonMatrix(matrix):
    detrendedAndDeseasoned = []
    for timeSeries in matrix:
        # Detrend
        trend = preprocessing.getTrend(timeSeries)
        detrended = preprocessing.removeTrend(timeSeries, trend)
        # Deseason the detrended data
        season = preprocessing.getSeasons(detrended)
        deseasoned = preprocessing.removeSeasons(timeSeries, season)
        # Add detrended and deseasoned data to list
        detrendedAndDeseasoned.append(deseasoned)
    return detrendedAndDeseasoned

def detrendAndDeseason(timeSeries):
    trend = preprocessing.getTrend(timeSeries)
    detrended = preprocessing.removeTrend(timeSeries, trend)
    # Deseason the detrended data
    season = preprocessing.getSeasons(detrended)
    deseasoned = preprocessing.removeSeasons(timeSeries, season)
    return deseasoned

def splitMultipleSeries(matrix, frameWidth):
    splitSeries = []
    for timeSeries in matrix:
        splitTimeSeries = preprocessing.splitSeries(timeSeries, frameWidth, 0)
        for series in splitTimeSeries:
            splitSeries.append(series)
    return splitSeries