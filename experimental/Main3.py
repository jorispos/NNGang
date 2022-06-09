# Yoo lads, I hope this code makes sense, I have tried my best to add as much documentation as possible
# Try to play around with Henk and get familiar with our soon-to-be super-intelligence
# if you have any questions feel free to let me know

# Import Libraries
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

# Load the datasets
file1 = open('Data/subset1.csv')
file2 = open('Data/subset2.csv')
file3 = open('Data/subset3.csv')
type(file1)
type(file2)
type(file3)

# Extract and store data from dataset 1
csvreader1 = csv.reader(file1)
rows1 = []
for row in csvreader1:
    intRow = [int(x) for x in row]
    rows1.append(intRow)

# Extract and store data from dataset 2
csvreader2 = csv.reader(file2)
rows2 = []
for row in csvreader2:
    intRow = [int(x) for x in row]
    rows2.append(intRow)

# Extract and store data from dataset 3
csvreader3 = csv.reader(file3)
rows3 = []
for row in csvreader3:
    intRow = [int(x) for x in row]
    rows3.append(intRow)

# Split up all data so that:
# X = [[1, 2, 3], || y = [4,
#      [6, 7, 8]] ||     [9]
X1 = []
y1 = []
# NOTE: have to manually set this variable to the length of the time series (68 in our case)
timeLength1 = len(rows1[0])
for row in rows1:
    X1.append(row[:timeLength1-1])
    y1.append(row[timeLength1-1])

# Dataset 2
X2 = []
y2 = []
# NOTE: have to manually set this variable to the length of the time series (68 in our case)
timeLength2 = len(rows2[0])
for row in rows2:
    X2.append(row[:timeLength2-1])
    y2.append(row[timeLength2-1])

# Dataset 3
X3 = []
y3 = []
# NOTE: have to manually set this variable to the length of the time series (68 in our case)
timeLength3 = len(rows3[0])
for row in rows3:
    X3.append(row[:timeLength3-1])
    y3.append(row[timeLength3-1])

# Length of the amount of data matrices (for x and y)
X1Length = len(y1)
X2Length = len(y2)
X3Length = len(y3)

# Set to the % of how much data should be used for training
trainingSplit = 0.9
trainingSize1 = round(X1Length*trainingSplit)
trainingSize2 = round(X2Length*trainingSplit)
trainingSize3 = round(X3Length*trainingSplit)

# Split the data into testing and training data (for cross-validation)
X1_train = []
X2_train = []
X3_train = []
X1_test = []
X2_test = []
X3_test = []
y1_train = []
y2_train = []
y3_train = []
y1_test = []
y2_test = []
y3_test = []

# Split the actual data into testing and training data
for rowIndex in range(X1Length):
    # Should the current row be used for training or test?
    if rowIndex < trainingSize1:
        # Use current row for training
        X1_train.append(X1[rowIndex])
        y1_train.append(y1[rowIndex])
    else:
        # Use current row for testing
        X1_test.append(X1[rowIndex])
        y1_test.append(y1[rowIndex])

# Dataset 2
for rowIndex in range(X2Length):
    # Should the current row be used for training or test?
    if rowIndex < trainingSize2:
        # Use current row for training
        X2_train.append(X2[rowIndex])
        y2_train.append(y2[rowIndex])
    else:
        # Use current row for testing
        X2_test.append(X2[rowIndex])
        y2_test.append(y2[rowIndex])

# Dataset 3
for rowIndex in range(X3Length):
    # Should the current row be used for training or test?
    if rowIndex < trainingSize3:
        # Use current row for training
        X3_train.append(X3[rowIndex])
        y3_train.append(y3[rowIndex])
    else:
        # Use current row for testing
        X3_test.append(X3[rowIndex])
        y3_test.append(y3[rowIndex])

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
henk.fit(X1_train, y1_train)

if False:
    # Predict using our multi-layer perceptron model.
    print("predictions:")
    predictedValues = henk.predict(X1_test)
    print(predictedValues)

    # Return the coefficient of determination of the prediction.
    print("score (R^2):")
    print(henk.score(X1_test, y1_test))

    # For illustrative purposes; plot the graph and prediction of the first prediction of the set
    # numPlots is set to the amount of graphs you want to produce
    numPlots = len(X1_test)

    for i in range(numPlots):
        plot = plt.figure(i)
        plt.plot(range(1, timeLength1, 1), X1_test[i], 'yellow')
        plt.plot(timeLength1, y1_test[i], 'yo')
        plt.plot(timeLength1, predictedValues[i], 'ro')
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
        print("x-hor length: " + str(len(X1[0])))
        print("x-ver length: " + str(len(X1)))
        print("y-hor length: " + str(len(y1)))
        print("x-hor test length: " + str(len(X1_test[0])))
        print("x-ver test length: " + str(len(X1_test)))
        print("y-hor test length: " + str(len(y1_test)))
        print("x-hor train length: " + str(len(X1_train[0])))
        print("x-ver train length: " + str(len(X1_train)))
        print("y-hor train length: " + str(len(y1_train)))
    if debugMatrices:
        print("x-array:")
        print(X1)
        print("y-array:")
        print(y1)
        print("x-train-array")
        print(X1_train)
        print("y-train-array")
        print(y1_train)
        print("x-test-array:")
        print(X1_test)
        print("y-test-array:")
        print(y1_test)