# Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import csv

# Load the dataset
file = open('Data/MockData.csv')
type(file)

# Extract and store data from dataset
csvreader = csv.reader(file)
rows = []
for row in csvreader:
    intRow = [int(x) for x in row]
    rows.append(intRow)

# Split up all data so that:
# X = [[1, 2, 3], || y = [4,
#      [6, 7, 8]] ||     [9]
X = []
y = []
timeLength = 68
for row in rows:
    X.append(row[:timeLength-1])
    y.append(row[timeLength-1])

XLength = len(y)

# Set to the % of how much data should be used for training
trainingSplit = 0.8
trainingSize = round(XLength*trainingSplit)

# Split the data into testing and training data (for cross-validation)
X_train = []
X_test = []
y_train = []
y_test = []

# Split the actual data
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

# Initialize the MLP Regressor
# Currently uses the same parameter as the one in the Nestor example report
regr = MLPRegressor(random_state=1,
                    max_iter=10000,
                    hidden_layer_sizes=(5, 5),
                    solver="lbfgs",
                    activation="relu",
                    tol=0.0001,
                    alpha=0.00001,
                    )

# Used for Debugging!
# Set the two variables to False/True for matrix debugging/visualization
debugMatrices = True
debugMatricesInfo = True
# Print the appropriate info
if debugMatricesInfo:
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
    print("x-hor length: " + str(len(X[0])))
    print("x-ver length: " + str(len(X)))
    print("y-hor length: " + str(len(y)))
    print("x-hor test length: " + str(len(X_test[0])))
    print("x-ver test length: " + str(len(X_test)))
    print("y-hor test length: " + str(len(y_test)))
    print("x-hor train length: " + str(len(X_train[0])))
    print("x-ver train length: " + str(len(X_train)))
    print("y-hor train length: " + str(len(y_train)))

# Train the MLPRegressor with the training data (X: input, y: output)
regr.fit(X_train, y_train)

# Return the coefficient of determination of the prediction.
print("score (R^2):")
print(regr.score(X_test, y_test))