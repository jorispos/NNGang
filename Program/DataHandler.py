import Program.Utils as utils

class DataHandler:

    matrix = []

    train = []
    test = []
    shown = []
    hidden = []

    def __init__(self, matrix):
        self.matrix = matrix

    # split all of the data in the points that we do know, and the points that were not known in the m3 competition
    def splitHidden(self, trainingPoints):
        for row in self.matrix:
            rowLength = len(row)
            self.shown.append(self.matrix[rowLength-trainingPoints:])
            self.hidden.append(self.matrix[:rowLength-trainingPoints])

    # split the known data, i.e. shown into the training data and the testing data(for cross-validation)
    def splitTrainTest(self, trainingSplit):
        matrixHeight = len(self.matrix)
        trainingSize = round(matrixHeight*trainingSplit)

        for rowIndex in range(matrixHeight):
            if rowIndex < trainingSize:
                # Use current row for training
                self.train.append(self.shown[rowIndex])
            else:
                # Use current row for testing
                self.test.append(self.shown[rowIndex])