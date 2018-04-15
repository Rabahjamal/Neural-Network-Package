import numpy as np
from matplotlib import pyplot as plt
from data_reader import Reader
from scipy._lib._ccallback_c import plus1_t


class Perceptron(object):
    def fit(self ,X, y, learning_rate, epochs):
        self.w_ = np.random.rand(X.shape[1] + 1)
        self.cost_ = []
        for i in range(epochs):
            v = self.__net_value(X)
            output = self.__signum(v)
            errors = y - output
            self.w_[1:] = self.w_[1:] + (learning_rate * X.T.dot(errors))
            self.w_[0] = self.w_[0] + learning_rate * errors.sum()
            self.cost_.append(errors.mean())


    def __net_value(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def __signum(self, v):
        return np.where(v > 0, 1, -1)

    def predict(self, x, y):
        net_value = self.__net_value(x)
        output = self.__signum(net_value)
        correct = 0
        for i in range(len(y)):
            if y[i] == output[i]:
                correct = correct + 1
        return (float(correct) / float(len(y))) * 100, output


#main
# Get the features to draw from the user
a = int(input("Please Enter the first feature you want to train the data on: "))
b = int(input("Please Enter the second feature you want to train the data on: "))

# Get the two classes from the user
class1 = int(input("Please Enter a number from 1 to 3: "))
class2 = int(input("Please Enter another number from 1 to 3: "))

x_train, x_test, y_train, y_test = Reader('data/Iris Data.txt').load_train_data(a, b, class1, class2)

# Mean normalization for data
mean1, mean2 = np.mean(x_train[:, 0]), np.mean(x_train[:, 1])
min1, max1 = x_train[:, 0].min(), x_train[:, 0].max()
min2, max2 = x_train[:, 1].min(), x_train[:, 1].max()
x_train[:, 0], x_train[:, 1] = (x_train[:, 0] - mean1) / (max1 - min1), (x_train[:, 1] - mean2) / (max2 - min2)
x_test[:, 0], x_test[:, 1] = (x_test[:, 0] - mean1) / (max1 - min1), (x_test[:, 1] - mean2) / (max2 - min2)

# Build the model
slp = Perceptron()
slp.fit(x_train, y_train, learning_rate=0.02, epochs=10)
acc, predicted = slp.predict(x_test, y_test)
print(acc)


# x1_min, x1_max = np.amin(x_train[:,0]), np.amax(x_train[:,1])
# x2_min, x2_max = np.amin(x_train[:,1]), np.amax(x_train[:,1])
# X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
# h = slp.w_[1] * X1 + slp.w_[2] * X2 + slp.w_[0] - 1
# plt.contour(X1, X2, h, [0], colors = 'k', linestyles = 'solid')
plt.plot(x_train[:30, 0], x_train[:30, 1], 'ro')
plt.plot(x_train[30:, 0], x_train[30:, 1], 'go')
#plt.plot(x_train[:, 0], np.dot(x_train, slp.w_[1:])+slp.w_[0], label = 'Fitted Line')
#plt.plot(slp.cost_)
plt.xlabel('X%s' %str(a))
plt.ylabel('X%s' %str(b))
plt.show()
# Plot the model

# plt.show()


