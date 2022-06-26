from sklearn.neural_network import MLPRegressor


class Henk:
    random_state = 1
    max_iter = 10000
    solver = "adam"
    activation = "tanh"
    tol = 0.00004
    alpha = 0.0005
    network = (5, 6, 7)

    def __init__(self):
        self.MLP = MLPRegressor(random_state=self.random_state,
                                max_iter=self.max_iter,
                                hidden_layer_sizes=self.network,
                                solver=self.solver,
                                activation=self.activation,
                                tol=self.tol,
                                alpha=self.alpha,
                                )