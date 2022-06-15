# Imports
import Program.Utils as utils
import Program.Preprocessing as preprocessing
from Program.DataHandler import DataHandler
from Program.Henk import Henk
from Program.Scaler import Scaler

# Constants
dataPath = '../Data/subset2.csv'
trainingSplit = 0.85
predictionPoints = 18

# Retrieve the time series matrix from the CSV data
print("loading data from drive..")
timeSeriesMatrix = utils.getRows(dataPath)

# Split the data into x and y and train and test
print("handling and splitting data..")
data = DataHandler(timeSeriesMatrix)
data.splitHidden(predictionPoints)
data.splitTrainTest(trainingSplit)

utils.printArray(data.train)
print("test")
utils.printArray(data.test)
print("hidden")
utils.printArray(data.hidden)

#first split the last 18 points from all data

#then on all data except the 18 points:
#split all the data in 14 X points and 1 Y point and move this 'frame'
#detrend, deseasonalize and scale the data
#store the trend season and scale for all the dataframes
    #since you need to add these back when you want to predict the split off 18 points in the end(prob store them in an array)

#then split the data into train and test
#fit Henk on the training data(without the 18 points), it learns by comparing its prediction to the actual value
#cross validate if Henk does well by using cross validation on the testing data(without the 18 points)

#now change the parameters according to the result of cross validation

#after you have set the hyperparameters to be good and checked these again:
#let Henk predict for all data the 18 points that you first splitted
#add to these the trend season and scale back
#compare the output to the actual output of the timeseries
#calculate the score of Henk, with smape etc