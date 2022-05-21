# Import Libraries
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import csv

# Load the dataset
file = open('Data/MockData.csv')
type(file)

# Extract data from dataset
csvreader = csv.reader(file)
xTime = next(csvreader)
yPrice = []
for row in csvreader:
    yPrice.append(row)

# Create the dataset
# X : the input samples
# y : the output samples
X, y = make_regression(n_samples=10, random_state=1)
# Split the dataset between training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Initialize the MLP Regressor
# Currently uses the same parameter as the one in the Nestor example report
regr = MLPRegressor(random_state=1,
                    max_iter=1000,
                    hidden_layer_sizes=(2, 1),
                    solver="lbfgs",
                    activation="relu",
                    tol=0.0001,
                    alpha=0.00001,
                    )

# Train the MLPRegressor with the training data (X: input, y: output)
regr.fit(X_train, y_train)

# Print data
print("overall data X:")
print(X)
print("overall data Y:")
print(y)
print("data to be predicted:")
print(X_test[:2])

print("x-hor length: " + str(len(X[0])))
print("x-ver length: " + str(len(X)))
print("y-hor length: " + str(len(y)))

print("x-hor test length: " + str(len(X_test[0])))
print("x-ver test length: " + str(len(X_test)))
print("y-hor test length: " + str(len(y_test)))

print("x-hor train length: " + str(len(X_train[0])))
print("x-ver train length: " + str(len(X_train)))
print("y-hor train length: " + str(len(y_train)))

# Predict using the multi-layer perceptron model.
print("prediction:")
print(regr.predict(X_test[:2]))

# Return the coefficient of determination of the prediction.
print("score (R^2):")
print(regr.score(X_test, y_test))