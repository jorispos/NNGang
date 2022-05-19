# Import Libraries
from sklearn.neural_network import MLPRegressor
import csv

# Load the dataset
file = open('Data/MockData.csv')
type(file)

# Extract and store data from dataset
csvreader = csv.reader(file)
xTime = next(csvreader)
yPrice = []
for row in csvreader:
    yPrice.append(row)

# Split the dataset between training and testing data
# X_train, X_test, y_train, y_test = train_test_split(xTime, yPrice, test_size=0.2, shuffle=False, stratify=None)

X_train = xTime[:60]
X_test = xTime[60:]
y_train = yPrice[:400]
y_test = yPrice[400:]

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

# Return the coefficient of determination of the prediction.
print("score (R^2):")
print(regr.score(X_test, y_test))