import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Read CSV file and return the integer matrix
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