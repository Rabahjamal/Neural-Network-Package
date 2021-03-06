import numpy as np
from matplotlib import pyplot as plt

class Reader(object):
    def __init__(self, file_path):
        self.__file_path = file_path

    def get_iris_data(self):
        features, labels = [], []

        with open(self.__file_path, 'r') as read:
            lines = read.readlines()

            for line in lines[1:]:
                tokens = line.strip().split(',')
                features.append(list(map(lambda v: float(v), [tokens[0] ,tokens[1], tokens[2], tokens[3]])))
                labels.append(str(tokens[4]))

        return features, labels

    def load_train_data(self):
        X, y = self.get_iris_data()
        x_train, x_test, y_train, y_test = [], [], [], []

        # train data
        x_train = [item for item in X[0:30]]
        x_train = x_train + [item for item in X[50:80]]
        x_train = x_train + [item for item in X[100:130]]
        y_train = [1 for i in range(30)]
        y_train = y_train + [2 for i in range(30)]
        y_train = y_train + [3 for i in range(30)]

        # test data
        x_test = [item for item in X[30:50]]
        x_test = x_test + [item for item in X[80:100]]
        x_test = x_test + [item for item in X[130:150]]
        y_test = [1 for i in range(20)]
        y_test = y_test + [2 for i in range(20)]
        y_test = y_test + [3 for i in range(20)]

        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    def helper(self, C1, C2, y1, y2):
        x_train, x_test, y_train, y_test = [], [], [], []

        for i in range(len(C1)):
            if i >= 0 and i < 30:
                x_train.append(C1[i])
            elif i >= 30 and i < 50:
                x_test.append(C1[i])

        for i in range(len(C2)):
            if i >= 0 and i < 30:
                x_train.append(C2[i])
            elif i >= 30 and i < 50:
                x_test.append(C2[i])

        for i in range(len(y1)):
            if i < 30:
                y_train.append(1)
            elif i >= 30 and i < 50:
                y_test.append(1)

        for i in range(len(y2)):
            if i < 30:
                y_train.append(-1)
            elif i >= 30 and i < 50:
                y_test.append(-1)

        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
