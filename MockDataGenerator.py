import csv

# open the file in the write mode
import random

f = open('Data/MockData.csv', 'w')

# create the csv writer
writer = csv.writer(f)

# write a row to the csv file

data = []
for col in range(500):
    randomStart = random.randint(1000, 10000)
    rows = []
    for row in range(68):
        rows.append(randomStart)
        randomStart += random.randint(8, 12)
    data.append(rows)

writer.writerows(data)

# close the file
f.close()