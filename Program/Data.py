class Data:
    X = []
    y = []
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    timeLength = 0
    trainingSplit = 0
    trainingSize = 0
    XLength = 0
    matrix = []

    def __init__(self, matrix, trainingSplit):
        self.trainingSplit = trainingSplit
        self.matrix = matrix

    def handle(self):
        self.splitXY()
        self.splitTrainTest()

    def splitXY(self):
        self.timeLength = len(self.matrix[0])
        for row in self.matrix:
            self.X.append(row[:self.timeLength-1])
            self.y.append(row[self.timeLength-1])
        self.XLength = len(self.y)
        self.trainingSize = round(self.XLength*self.trainingSplit)

    def splitTrainTest(self):
        for rowIndex in range(self.XLength):
            if rowIndex < self.trainingSize:
                # Use current row for training
                self.X_train.append(self.X[rowIndex])
                self.y_train.append(self.y[rowIndex])
            else:
                # Use current row for testing
                self.X_test.append(self.X[rowIndex])
                self.y_test.append(self.y[rowIndex])