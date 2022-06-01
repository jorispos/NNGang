# Yoo lads, I hope this code makes sense, I have tried my best to add as much documentation as possible
# Try to play around with Henk and get familiar with our soon-to-be super-intelligence
# if you have any questions feel free to let me know

# Import Libraries
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import meegkit
from meegkit.detrend import regress, detrend
import numpy as np
from matplotlib import mlab
import scipy


# Load the dataset
file = open('Data/subset2.csv')
type(file)

# Extract and store data from dataset
csvreader = csv.reader(file)
rows = []
for row in csvreader:
    intRow = [int(x) for x in row]
    rows.append(intRow)

#np.array(rows)

# y, _, = meegkit.detrend.detrend(
#       np.array(rows)
#     , w=None
#     , order=2
#     , basis='polynomials'
#     , threshold=3
#     , n_iter=4
#     , show=True)

#y, _, _ = detrend(np.array(rows), 2)


#print(mlab.detrend_linear(rows))

rows = scipy.signal.detrend(rows, axis=- 1, type='linear', bp=0, overwrite_data=False)

# Split up all data so that:
# X = [[1, 2, 3], || y = [4,
#      [6, 7, 8]] ||     [9]
X = []
y = []
# NOTE: have to manually set this variable to the length of the time series (68 in our case)
timeLength = len(rows[0])
for row in rows:
    X.append(row[:timeLength-1])
    y.append(row[timeLength-1])

# Length of the amount of data matrices (for x and y)
XLength = len(y)

# Set to the % of how much data should be used for training
trainingSplit = 0.9
trainingSize = round(XLength*trainingSplit)

# Split the data into testing and training data (for cross-validation)
X_train = []
X_test = []
y_train = []
y_test = []

# Split the actual data into testing and training data
for rowIndex in range(XLength):
    # Should the current row be used for training or test?
    if rowIndex < trainingSize:
        # Use current row for training
        X_train.append(X[rowIndex])
        y_train.append(y[rowIndex])
    else:
        # Use current row for testing
        X_test.append(X[rowIndex])
        y_test.append(y[rowIndex])

# Initialize our MLP Regressor, Henk
# Currently uses the same parameter as the one in the Nestor example report (except for the hidden layer size)
henk = MLPRegressor(random_state=1,
                    max_iter=10000,
                    hidden_layer_sizes=(5, 5),
                    solver="lbfgs",
                    activation="relu",
                    tol=0.0001,
                    alpha=0.00001,
                    )

# Train the MLPRegressor with the training data (X: input, y: output)
henk.fit(X_train, y_train)

# Predict using our multi-layer perceptron model.
print("predictions:")
predictedValues = henk.predict(X_test)
print(predictedValues)

# Return the coefficient of determination of the prediction.
print("score (R^2):")
print(henk.score(X_test, y_test))

# For illustrative purposes; plot the graph and prediction of the first prediction of the set
# numPlots is set to the amount of graphs you want to produce
numPlots = len(X_test)

for i in range(numPlots):
    plot = plt.figure(i)
    plt.plot(range(1, timeLength, 1), X_test[i], 'yellow')
    plt.plot(timeLength, y_test[i], 'yo')
    plt.plot(timeLength, predictedValues[i], 'ro')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Henk in action')
    redLegend = mpatches.Patch(color='red', label='Predicted data')
    yellowLegend = mpatches.Patch(color='#D5CF0C', label='Actual data')
    plt.legend(handles=[redLegend, yellowLegend])
    plt.show()

# Used for Debugging! (can ignore for now, or play around with it)
# Set the two variables to False/True for matrix debugging/visualization
debugMatrices = False
debugMatricesInfo = False
# Print the appropriate info
if debugMatricesInfo:
    print("x-hor length: " + str(len(X[0])))
    print("x-ver length: " + str(len(X)))
    print("y-hor length: " + str(len(y)))
    print("x-hor test length: " + str(len(X_test[0])))
    print("x-ver test length: " + str(len(X_test)))
    print("y-hor test length: " + str(len(y_test)))
    print("x-hor train length: " + str(len(X_train[0])))
    print("x-ver train length: " + str(len(X_train)))
    print("y-hor train length: " + str(len(y_train)))
if debugMatrices:
    print("x-array:")
    print(X)
    print("y-array:")
    print(y)
    print("x-train-array")
    print(X_train)
    print("y-train-array")
    print(y_train)
    print("x-test-array:")
    print(X_test)
    print("y-test-array:")
    print(y_test)