import csv
import random

startRange = 1000
endRange = 10000

fluctuationStart = -15
fluctuationEnd = 20

numRows = 500
numCols = 68

with open('Data/MockData.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    data = []

    for col in range(numRows):
        randomStart = random.randint(startRange, endRange)
        rows = []
        for row in range(numCols):
            rows.append(randomStart)
            randomStart += random.randint(fluctuationStart, fluctuationEnd)
        data.append(rows)

    writer.writerows(data)

print("New .csv file successfully created :)")