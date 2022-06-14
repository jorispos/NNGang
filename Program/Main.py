# Imports
from matplotlib import pyplot as plt

import Program.Utils as utils
import Program.Preprocessing as preprocessing
from Program.Data import Data
from Program.Henk import Henk

# Config constants
from Program.Scaler import Scaler

dataPath = '../Data/subset2.csv'
trainingSplit = 0.9

# Retrieve the time series matrix from the CSV data
timeSeriesMatrix = utils.getRows(dataPath)

# Split the data into x and y and train and test
data = Data(timeSeriesMatrix, trainingSplit)
data.handle()

# Create our MLP
henk = Henk()
# Train Henk
henk.MLP.fit(data.X_train, data.y_train)

# Make predictions on testing data and score the algorithm
predictedValues = henk.MLP.predict(data.X_test)
print(predictedValues)
score = henk.MLP.score(data.X_test, data.y_test)
print(score)

# Visually display the graphs
# utils.displayGraphs(data.X_test, data.y_test, predictedValues, data.timeLength)

# Experimental/Debugging for testing
deseasoned = []
for i in range(10):
    trend = preprocessing.getTrend(timeSeriesMatrix[i])
    detrend = preprocessing.removeTrend(timeSeriesMatrix[i], trend)
    season = preprocessing.getSeasons(detrend)
    deseason = preprocessing.removeSeasons(detrend, season)
    deseasoned.append(deseason)
    plt.plot(deseason)
    plt.title('Deseasoned ' + str(i), fontsize=16)
    plt.show()

# Scale the data
scaler = Scaler()
scaler.fit(deseasoned)
scaledData = scaler.transform(deseasoned)

for i in range(len(scaledData)):
    plt.plot(scaledData[i])
    plt.title('Scaled ' + str(i), fontsize=16)
    plt.show()

# StandardScaler
# scaler = StandardScaler()
# scaler.fit(scaleData)
# scaledData = scaler.transform(scaleData)

print("program finished :)")