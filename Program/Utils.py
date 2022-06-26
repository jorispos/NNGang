import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

import Program.Preprocessing as preprocessing
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
        deseasoned = preprocessing.removeSeasons(detrended, season)
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


def getTrends(matrix):
    trends = []
    for row in matrix:
        trends.append(preprocessing.getTrend(row))
    return trends


def getDetrendedSeasons(matrix):
    seasons = []
    for row in matrix:
        row = preprocessing.removeTrend(row, preprocessing.getTrend(row))
        seasons.append(preprocessing.getSeasons(row))
    return seasons


def getLastFrames(matrix, frameWidth):
    startingFrames = []
    for row in matrix:
        startingFrames.append(row[len(row) - frameWidth:])
    return startingFrames


def getFirstFrames(matrix, frameWidth):
    frames = []
    for row in matrix:
        frames.append(row[:frameWidth])
    return frames


def getLastItems(matrix):
    array = []
    for row in matrix:
        array.append(row[len(row) - 1])
    return array


def appendRows(matrix1, matrix2):
    newMatrix = []
    for i in range(len(matrix1)):
        newMatrix.append(numpy.append(matrix1[i], matrix2[i]))
    return newMatrix


def duplicateMatrix(matrix):
    newMatrix = []
    for row in matrix:
        newMatrix.append(row)
    return newMatrix


def addArrays(array1, array2):
    newMatrix = []
    for i in range(len(array1)):
        newMatrix.append(preprocessing.addArray(array1[i], array2[i]))
    return newMatrix


def popFirst(array):
    newMatrix = []
    for row in array:
        newMatrix.append(numpy.delete(row, 0))
    return newMatrix


def getFrames(matrix, start, end):
    newMatrix = []
    for i in range(len(matrix)):
        newMatrix.append(matrix[i][start:end])
    return newMatrix


def matrixToCsv(matrix, path):
    with open(path, 'w', newline='') as file:
        mywriter = csv.writer(file, delimiter=',')
        mywriter.writerows(matrix)


def printOverlayMatrices(matrix1, matrix2):
    for i in range(len(matrix1)):
        plt.plot(matrix1[i], c="blue")
        plt.plot(matrix2[i], c="red")
        plt.title("Graph " + str(i), fontsize=16)
        plt.show()


def graphPredictionsOverlay(timeSeries, predictions, dir):
    predictionPoints = len(predictions)
    timeSeriesLength = len(timeSeries)
    plt.clf()

    plt.plot(range(0, timeSeriesLength-predictionPoints), timeSeries[0:timeSeriesLength-predictionPoints], c = "lawngreen")
    plt.plot(range(timeSeriesLength-predictionPoints-1, timeSeriesLength), timeSeries[timeSeriesLength-predictionPoints-1:], c = "greenyellow")
    plt.plot(range(timeSeriesLength-predictionPoints-1, timeSeriesLength), numpy.append(timeSeries[timeSeriesLength-predictionPoints-1], predictions), c = "orange")
    plt.xlabel('Time (days)')
    plt.ylabel('Price ($)')
    plt.title('Henk in action')
    shownLegend = mpatches.Patch(color='lawngreen', label='Given Data')
    predictedLegend = mpatches.Patch(color='greenyellow', label='Actual Events')
    actualLegend = mpatches.Patch(color='orange', label='Predicted Data')
    plt.legend(handles=[shownLegend, predictedLegend, actualLegend])
    plt.show()
    #plt.savefig(dir)

def graphPredictionsOverlayMatrix(timeSeriesMatrix, predictionsMatrix, graphs, dir):
    print("Generating " + str(len(timeSeriesMatrix)) + " plots..")
    for i in range(graphs):
        graphPredictionsOverlay(timeSeriesMatrix[i], predictionsMatrix[i], dir + 'Graph' + str(i))
    print("Plots sucessfully generated and saved to: " + dir)

def calculateSmapeVector2(predictionData, actualData):
    differenceSum = 0
    for i in range(len(predictionData)):
        A = predictionData[i]
        F = predictionData[i]

        diff = abs(A - F)
        diff = diff / ((A + F)/2)
        differenceSum += diff
    differenceSum *= 100/len(predictionData)
    return differenceSum

def calculateSmapeVector(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def squeezeMatrix(matrix):
    newMatrix = []
    for row in matrix:
        for item in row:
            newMatrix.append(item)
    return np.array(newMatrix)

def calculateSmapeMatrix(matrix1, matrix2):
    vector1 = squeezeMatrix(matrix1)
    vector2 = squeezeMatrix(matrix2)
    return calculateSmapeVector(vector1, vector2)

def getTrendModels(matrix):
    models = []

    for timeSeries in matrix:
        # Reformat X and y
        X = range(len(timeSeries))
        X = np.reshape(X, (len(X), 1))
        y = timeSeries

        # Compute linear model
        pf1 = PolynomialFeatures(degree=1)
        Xp1 = pf1.fit_transform(X)
        md1 = LinearRegression()
        md1.fit(Xp1, y)

        # Compute quadratic model
        pf2 = PolynomialFeatures(degree=2)
        Xp2 = pf2.fit_transform(X)
        md2 = LinearRegression()
        md2.fit(Xp2, y)
        models.append((md1, md2))

    return models

def getTrendsFromModels(matrix, models):
    trendPoints = []

    for i in range(len(matrix)):
        Xp1 = range(len(matrix[i]) + 18)
        Xp1 = np.reshape(Xp1, (len(Xp1), 1))
        pf1 = PolynomialFeatures(degree=1)
        Xp1 = pf1.fit_transform(Xp1)
        points1 = models[i][0].predict(Xp1)

        Xp2 = range(len(matrix[i]) + 18)
        Xp2 = np.reshape(Xp2, (len(Xp2), 1))
        pf2 = PolynomialFeatures(degree=2)
        Xp2 = pf2.fit_transform(Xp2)
        points2 = models[i][1].predict(Xp2)

        trend = [(points1[i] + points2[i])/2 for i in range(0, len(points1))]
        trendPoints.append(trend)

    return trendPoints


def getEntireSeasons(seasons, data):
    newMatrix = []

    for i in range(len(seasons)):
        preferredLength = len(data[i])
        seasonality = seasons[i]

        season = []
        replications = (int) ((preferredLength / len(seasonality))+1)
        for rep in range(replications):
            for item in seasonality:
                season.append(item)

        newMatrix.append(season[:preferredLength])
    return newMatrix