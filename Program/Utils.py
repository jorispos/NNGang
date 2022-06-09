import csv


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

def printArray(array):