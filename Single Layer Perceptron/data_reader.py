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

    def load_train_data(self, a, b, c1, c2):
        X, y = self.get_iris_data()

        # Splitting the date besed on the features entered by the user
        C1 = np.array([[item[a - 1], item[b - 1]] for item in X[:50]])
        C2 = np.array([[item[a - 1], item[b - 1]] for item in X[50:100]])
        C3 = np.array([[item[a - 1], item[b - 1]] for item in X[100:]])

        # Plotting the Iris dataset
        plt.plot(C1[:, 0], C1[:, 1], 'ro')
        plt.plot(C2[:, 0], C2[:, 1], 'go')
        plt.plot(C3[:, 0], C3[:, 1], 'bo')
        plt.xlabel('X%s' % str(a))
        plt.ylabel('X%s' % str(b))
        plt.show()

        if c1 == 1 and c2 == 2:
            x_train, x_test, y_train, y_test = self.helper(C1, C2, y[:50], y[50:100])
        elif c1 == 1 and c2 == 3:
            x_train, x_test, y_train, y_test = self.helper(C1, C3, y[:50], y[100:])
        else:
            x_train, x_test, y_train, y_test = self.helper(C2, C3, y[50:100], y[100:])

        return x_train, x_test, y_train, y_test

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
