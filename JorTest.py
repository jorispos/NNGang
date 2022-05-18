import numpy as np
from sklearn.neural_network import MLPRegressor

INPUTS = 4
HIDDEN_LAYER = 2
OUTPUTS = 6

# create the initial MLP:
mlp = MLPRegressor(hidden_layer_sizes=(HIDDEN_LAYER,), max_iter=1)

# This will initialize input and output layers, and nodes weights and biases:
# we are not otherwise interested in training the MLP here, hence the settings max_iter=1 above
mlp.fit(np.random.uniform(low=-1, high=1, size=INPUTS).reshape(1, -1), np.ones(OUTPUTS))

# weights are represented as a list of 2 ndarrays:
# - hidden layer weights: INPUTS x HIDDEN_LAYER
# - output layer weights: HIDDEN_LAYER x OUTPUTS
numWeights = INPUTS * HIDDEN_LAYER + HIDDEN_LAYER * OUTPUTS

#Setup Network Parameters
netParams = []
netParams.append(range(0, numWeights, 0))
netParams.append(range(0, numWeights, 0))

weights = np.array(netParams[:numWeights])
mlp.coefs_ = [
    weights[0:INPUTS * HIDDEN_LAYER].reshape((INPUTS, HIDDEN_LAYER)),
    weights[INPUTS * HIDDEN_LAYER:].reshape((HIDDEN_LAYER, OUTPUTS))
]

# biases are represented as a list of 2 ndarrays:
# - hidden layer biases: HIDDEN_LAYER x 1
# - output layer biases: OUTPUTS x 1
biases = np.array(netParams[numWeights:])
mlp.intercepts_ = [biases[:HIDDEN_LAYER], biases[HIDDEN_LAYER:]]