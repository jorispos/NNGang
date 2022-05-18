# Import Libraries
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Create the dataset
# X : the input samples
# y : the output samples
X, y = make_regression(n_samples=10, random_state=1)
# Split the dataset between training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Initialize the MLP Regressor
# Parameters inspired by report found on nestor
regr = MLPRegressor(random_state=1,
                    max_iter=5000,
                    hidden_layer_sizes=(2, 1),
                    solver="lbfgs",
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

# Predict using the multi-layer perceptron model.
print("prediction:")
print(regr.predict(X_test[:2]))

# Return the coefficient of determination of the prediction.
print("score (R^2):")
print(regr.score(X_test, y_test))