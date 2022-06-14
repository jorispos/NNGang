from sklearn.preprocessing import MinMaxScaler


class Scaler:

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(-1,1))

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)