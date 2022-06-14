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
trend1 = preprocessing.getTrend(timeSeriesMatrix[0])
detrend1 = preprocessing.removeTrend(timeSeriesMatrix[0], trend1)
season1 = preprocessing.getSeasons(detrend1)
deseason1 = preprocessing.removeSeasons(detrend1, season1)

trend2 = preprocessing.getTrend(timeSeriesMatrix[1])
detrend2 = preprocessing.removeTrend(timeSeriesMatrix[1], trend2)
season2 = preprocessing.getSeasons(detrend2)
deseason2 = preprocessing.removeSeasons(detrend2, season2)

trend3 = preprocessing.getTrend(timeSeriesMatrix[2])
detrend3 = preprocessing.removeTrend(timeSeriesMatrix[2], trend3)
season3 = preprocessing.getSeasons(detrend3)
deseason3 = preprocessing.removeSeasons(detrend3, season3)

scaleData = [deseason1, deseason2, deseason3]

# MinMaxScaler
scaler = Scaler()
scaler.fit(scaleData)
scaledData = scaler.transform(scaleData)

# StandardScaler
# scaler = StandardScaler()
# scaler.fit(scaleData)
# scaledData = scaler.transform(scaleData)

# Display matrix[0] process
# plt.plot(timeSeriesMatrix[0])
# plt.title('Original', fontsize=16)
# plt.show()
# plt.plot(detrend1)
# plt.title('Detrended', fontsize=16)
# plt.show()
plt.plot(deseason1)
plt.title('Deseasoned', fontsize=16)
plt.show()
plt.plot(scaledData[0])
plt.title('Scaled', fontsize=16)
plt.show()
plt.plot(deseason2)
plt.title('Deseasoned 2', fontsize=16)
plt.show()
plt.plot(scaledData[2])
plt.title('Scaled 2', fontsize=16)
plt.show()

print("program finished :)")