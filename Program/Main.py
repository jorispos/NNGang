# Imports
import Program.Utils as utils
import Program.Preprocessing as preprocessing
from Program.Data import Data
from Program.Henk import Henk
from Program.Scaler import Scaler

# Constants
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
preprocessed = []
numTests = 20

for i in range(numTests):
    trend = preprocessing.getTrend(timeSeriesMatrix[i])
    detrend = preprocessing.removeTrend(timeSeriesMatrix[i], trend)
    season = preprocessing.getSeasons(detrend)
    deseason = preprocessing.removeSeasons(detrend, season)
    preprocessed.append(deseason)
    utils.plotData(deseason, 'Preprocessed ' + str(i), 16)

# Scale the data
scaler = Scaler()
scaler.fit(preprocessed)
scaledData = scaler.transform(preprocessed)
utils.plotDataMatrix(scaledData, 'Scaled ', 16)

# StandardScaler
# scaler = StandardScaler()
# scaler.fit(scaleData)
# scaledData = scaler.transform(scaleData)

print("program finished :)")