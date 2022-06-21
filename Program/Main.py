# Imports
import Program.Utils as utils
from Program.Data import Data
from Program.Henk import Henk
from Program.Scaler import Scaler

# Constants
dataPath = '../Data/subset2.csv'
outputPath = '../Data/predictions.csv'
plotsPath = '../Plots/'
trainingSplit = 0.85
predictionPoints = 18
frameWidth = 15

# Retrieve the time series matrix from the CSV data
print("Loading data from drive..")
timeSeriesMatrix = utils.getRows(dataPath)

# Split the data into train, test, and hidden
print("Handling and splitting data..")
data = Data(timeSeriesMatrix)
data.splitHidden(predictionPoints)
data.splitTrainTest(trainingSplit)

# ----------------------
#     PREPROCESSING
# ----------------------

rawTrainingData = data.train
rawTestingData = data.test

# ---> Detrend, Deseasonalize both training and testing data
print("Detrending and deseasoning training data..")
detrendedDeseasonedTrainingData = utils.detrendAndDeseasonMatrix(rawTrainingData)
print("Detrending and deseasoning testing data..")
detrendedDeseasonedTestingData = utils.detrendAndDeseasonMatrix(rawTestingData)

# ---> Split all the detrended deseasoned time series into many small time-series of width frameWidth
print("Generating sub-series from training and testing timeseries..")
splitTrainTimeSeries = utils.splitMultipleSeries(detrendedDeseasonedTrainingData, 15)
splitTestTimeSeries = utils.splitMultipleSeries(detrendedDeseasonedTestingData, 15)

# ---> Scale the detrended and deseasoned testing & training data
print("Scaling training and testing data..")
scaler = Scaler()
scaler.fit(splitTrainTimeSeries + splitTestTimeSeries)
preprocessedTrainingData = scaler.transform(splitTrainTimeSeries)
preprocessedTestingData = scaler.transform(splitTestTimeSeries)
preprocessedData = utils.mergeArrays(preprocessedTrainingData, preprocessedTestingData)
print("Preprocessing successfully completed..")

# ----------------------
#        TRAINING
# ----------------------

# ---> Preparing training data for Henk
trainingData = Data(preprocessedTrainingData)
trainingData.splitXY()

# ---> Initializing Henk
print("Initializing Henk..")
henk = Henk()
print("Training Henk..")
henk.MLP.fit(trainingData.x, trainingData.y)
print("Successfully trained Henk on " + str(len(trainingData.x)) + " data samples..")

# ----------------------
#    CROSS-VALIDATING
# ----------------------

# ---> Preparing testing data for Henk
testingData = Data(preprocessedTestingData)
testingData.splitXY()

# ---> Score Henk
print("Cross-validating Henk on testing data..")
score = henk.MLP.score(testingData.x, testingData.y)
print("Henk R^2 score: " + str(score))

# -------------------------------
#    MAKE CONTEST PREDICTIONS
# -------------------------------

# ---> Get the Trends and Seasons of all Timeseries (used for preprocessing, not training)
print("Getting Henk ready for competition..")
trends = utils.getTrends(data.matrix)
trends = utils.getLastFrames(trends, predictionPoints + 14)
seasons = utils.getDetrendedSeasons(data.matrix)
seasons = utils.getLastFrames(seasons, predictionPoints + 14)

# ---> Get the last 15 points of each shown timeseries and scale (Henk will predict from here)
preprocessedShown = utils.detrendAndDeseasonMatrix(data.shown)
startingFrames = utils.getLastFrames(preprocessedShown, 15)
startingFrames = scaler.transform(startingFrames)
startingFrames = utils.getLastFrames(startingFrames, 14)

# ---> Start predicting
print("Henk is starting the competition..")
predictedValues = []
# Make 18 predictions
for i in range(predictionPoints):
    # Predict the next value for all rows
    rawPredictions = henk.MLP.predict(startingFrames)
    # Add predicted values to the end of all rows
    startingFrames = utils.appendRows(startingFrames, rawPredictions)
    # Create copy of the current frames to undo preprocessing
    frames = utils.duplicateMatrix(startingFrames)
    frames = scaler.scaler.inverse_transform(frames)
    framesSeasons = utils.getFrames(seasons, i, i+15)
    framesTrends = utils.getFrames(trends, i, i+15)
    frames = utils.addArrays(frames, framesSeasons)
    frames = utils.addArrays(frames, framesTrends)
    predictions = utils.getLastItems(frames)
    predictedValues.append(predictions)
    # Pop first element of startingFrames to prepare for next prediction
    startingFrames = utils.popFirst(startingFrames)

predictedValues = utils.transpose(predictedValues)

# ---> Save predictions
print("Henk has successfully made " + str(len(predictedValues[0])) + " predictions for "
      + str(len(predictedValues)) + " timeseries..")
utils.matrixToCsv(predictedValues, outputPath)
print("Predictions saved to: " + outputPath + "..")

# -----------------------------------------
#    QUANTITATIVE/QUALITATIVE ANALYSIS
# -----------------------------------------

# ---> Program finished
utils.graphPredictionsOverlayMatrix(data.matrix, predictedValues, len(data.matrix), plotsPath)
print("Program finished :)")