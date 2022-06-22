import Program.Utils as utils
from Program.Scaler import Scaler
# Constants
dataPath = '../Data/subset2.csv'

# Retrieve the time series matrix from the CSV data
print("Loading data from drive..")
timeSeriesMatrix = utils.getRows(dataPath)

scaler = Scaler()
scaler.fit(timeSeriesMatrix)

seasons = utils.getDetrendedSeasons(timeSeriesMatrix)

# ---> Preprocessing

# Original
print(timeSeriesMatrix[0])
utils.plotData(timeSeriesMatrix[0], "original", 16)

# Detrend and deseason
trends = utils.getTrends(timeSeriesMatrix)
timeSeriesMatrix = utils.detrendAndDeseasonMatrix(timeSeriesMatrix)
utils.plotData(timeSeriesMatrix[0], "detrend and deseasoned", 16)

# Scale
timeSeriesMatrix = scaler.transform(timeSeriesMatrix)
utils.plotData(timeSeriesMatrix[0], "scaled", 16)

# ---> Undo preprocessing

# Descale
timeSeriesMatrix = scaler.scaler.inverse_transform(timeSeriesMatrix)
utils.plotData(timeSeriesMatrix[0], "unscaled", 16)

# add seasons
timeSeriesMatrix = utils.addArrays(timeSeriesMatrix, seasons)
# add trends
timeSeriesMatrix = utils.addArrays(timeSeriesMatrix, trends)
utils.plotData(timeSeriesMatrix[0], "should be og", 16)

print(timeSeriesMatrix[0])