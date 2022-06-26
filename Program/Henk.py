from sklearn.neural_network import MLPRegressor


class Henk:
    random_state = 1
    max_iter = 10000
    solver = "lbfgs"
    activation = "relu"
    tol = 0.0001
    alpha = 0.00001

    def __init__(self, params):
        self.MLP = MLPRegressor(random_state=self.random_state,
                                max_iter=self.max_iter,
                                hidden_layer_sizes=params,
                                solver=self.solver,
                                activation=self.activation,
                                tol=self.tol,
                                alpha=self.alpha,
                                )