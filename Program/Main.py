# Imports
import Program.Utils as utils
import Program.Preprocessing as preprocessing
from Program.Data import Data
from Program.Henk import Henk
from Program.Scaler import Scaler

# Constants
dataPath = '../Data/subset2.csv'
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
trends = []
detrendedSeasons = []
for timeSeries in data.matrix:
    trend = preprocessing.getTrend(timeSeries)
    detrended = preprocessing.removeTrend(timeSeries, trend)
    trends.append(trend)
    season = preprocessing.getSeasons(detrended)
    detrendedSeasons.append(season)

# ---> Preprocess shown data and get last 15 points of each row for starting frames used to predict
preprocessedShown = utils.detrendAndDeseasonMatrix(data.shown)
# Scaler can only take matrix width 15
last15PreprocessedShown = []
for row in preprocessedShown:
    last15PreprocessedShown.append(row[len(row) - 15:])
# Scale the last 15 points of each shown timeSeries
last15PreprocessedScaledShown = scaler.transform(last15PreprocessedShown)

# Get the starting frames (last 14 frames of visible preprocessed data)
startingFrames = []
for timeSeries in last15PreprocessedScaledShown:
    startingFrame = timeSeries[len(timeSeries) - frameWidth + 1:]
    startingFrames.append(startingFrame)

# Iteratively make 18 predictions after each starting frame
print("Henk is starting the competition..")
contestPredictions = []
for i in range(predictionPoints):
    # Make predictions for all starting frames
    predictions = henk.MLP.predict(startingFrames)
    contestPredictions.append(predictions)
    # Add these predictions to all starting frames and shift the frame over one to the right

    # Add predictions to startingFrames
    # Descale, add season, add trend, save those predictions

    startingFrames = utils.addAndShift(startingFrames, predictions)

# Transpose to return to original format || Assuming that this works
contestPredictions = utils.transpose(contestPredictions)

# Undo scaling
# contestPredictions = scaler.scaler.inverse_transform(contestPredictions)

# first split the last 18 points from all data

# then on all data except the 18 points:
# split all the data in 14 X points and 1 Y point and move this 'frame'
# detrend, deseasonalize and scale the data
# store the trend season and scale for all the dataframes
# since you need to add these back when you want to predict the split off 18 points
# in the end(prob store them in an array)

# then split the data into train and test
# fit Henk on the training data(without the 18 points), it learns by comparing its prediction to the actual value
# cross validate if Henk does well by using cross validation on the testing data(without the 18 points)

# now change the parameters according to the result of cross validation

# after you have set the hyperparameters to be good and checked these again:
# let Henk predict for all data the 18 points that you first split
# add to these the trend season and scale back
# compare the output to the actual output of the timeseries
# calculate the score of Henk, with SMAPE etc