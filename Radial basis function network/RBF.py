import numpy as np
from data_reader import Reader
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

class RBF:
    def __init__(self):
        self.cost = []
        self.epochs = []
        self.centers = np.zeros(shape=(5, 4))
        self.w = np.random.rand(6, 3)

    def K_mean(self, X):
        centers = np.zeros(shape=(5, 4))
        centers[0], centers[1], centers[2], centers[3], centers[4] = X[0], X[18], X[36], X[54], X[72]
        dictionary = {}
        counter, iterations = 0, 0

        while counter < 5:
            # calculate the distance from each sample to each center
            for sample_id in range(X.shape[0]):
                min = 100000
                sample = X[sample_id]
                for center_id in range(centers.shape[0]):
                    center = centers[center_id]
                    dist = (sample[0] - center[0])*(sample[0] - center[0]) + (sample[1] - center[1])*(sample[1] - center[1]) + (sample[2] - center[2])*(sample[2] - center[2]) + (sample[3] - center[3])*(sample[3] - center[3])
                    dist = np.sqrt(dist)
                    if dist < min:
                        min = dist
                        dictionary[sample_id] = center_id

            # calculate the new centers
            counter = 0
            for center_id in range(5):
                center = centers[center_id]
                sum1, sum2, sum3, sum4, num_of_samples = 0, 0, 0, 0, 0
                for sample_id in range(X.shape[0]):
                    sample = X[sample_id]
                    if dictionary[sample_id] == center_id:
                        sum1, sum2, sum3, sum4 = sum1+sample[0], sum2+sample[1], sum3+sample[2], sum4+sample[3]
                        num_of_samples = num_of_samples + 1
                new_center = np.array([sum1/float(num_of_samples), sum2/float(num_of_samples), sum3/float(num_of_samples), sum4/float(num_of_samples)])
                if np.all(np.equal(new_center, center)):
                    counter += 1
                centers[center_id] = new_center

            k_mean_cost = 0
            for key in dictionary:
                sample = X[key]
                center = centers[dictionary[key]]
                dist = (sample[0] - center[0]) * (sample[0] - center[0]) + (sample[1] - center[1]) * (sample[1] - center[1]) + (sample[2] - center[2]) * (sample[2] - center[2]) + (sample[3] - center[3]) * (sample[3] - center[3])
                k_mean_cost += dist
            print("Iteration number ", iterations, ": Cost = ", k_mean_cost)
            iterations += 1
        print(dictionary)
        return centers

    def __net_value(self, a):
        return np.add(np.dot(a, self.w[1:]), self.w[0])

    def fit(self, X, target, epochs, learning_rate):
        centers = self.K_mean(X)
        self.centers = centers
        for epoch in range(epochs):
            list_of_h = []
            # Train the network
            for sample_id in range(X.shape[0]):
                sample = X[sample_id]
                # Input to hidden
                h = np.zeros(shape=centers.shape[0])
                i = 0
                for center in centers:
                    dist = (sample[0] - center[0]) * (sample[0] - center[0]) + (sample[1] - center[1]) * (sample[1] - center[1]) + (sample[2] - center[2]) * (sample[2] - center[2]) + (sample[3] - center[3]) * (sample[3] - center[3])
                    h[i] = np.exp((-1*dist) / 2.0)
                    i = i + 1
                list_of_h.append(h)

                # Hidden to Output
                net = self.__net_value(h)
                y = net
                error = target[sample_id] - y
                error = np.reshape(error, [1, error.shape[0]])
                h = np.reshape(h, [h.shape[0], 1])
                self.w[1:] = self.w[1:] + (learning_rate * np.dot(h, error))
                self.w[0] = self.w[0] + learning_rate * error

            # MSE Computation
            errors = []
            for i in range(len(list_of_h)):
                h = list_of_h[i]
                net = self.__net_value(h)
                y = net
                error = target[i] - y
                errors.append(np.mean(np.square(error)))

            errors = np.array(errors)
            mse = 0.5 * (np.mean(errors))
            self.cost.append(mse)
            self.epochs.append(epoch)
            print("Epoch: ", epoch, ", MSE = ", mse)

    def predict(self, X):
        output = []
        for sample_id in range(X.shape[0]):
            sample = X[sample_id]
            # Input to hidden
            h = np.zeros(shape=self.centers.shape[0])
            i = 0
            for center in self.centers:
                dist = (sample[0] - center[0]) * (sample[0] - center[0]) + (sample[1] - center[1]) * (sample[1] - center[1]) + (sample[2] - center[2]) * (sample[2] - center[2]) + (sample[3] - center[3]) * (sample[3] - center[3])
                h[i] = np.exp((-1 * dist) / 2.0)
                i = i + 1

            # Hidden to Output
            net = self.__net_value(h)
            y = net
            output.append(y)
        output = np.array(output)
        predictions = np.argmax(output, 1)
        return predictions

    def eval(self, pred, true):
        classes = np.argmax(true, 1)
        num_of_corrects = 0
        for i in range(len(classes)):
            if classes[i] == pred[i]:
                num_of_corrects = num_of_corrects + 1
        return (float(num_of_corrects)/float(len(classes))) * 100.0

# main
x_train, x_test, y_train, y_test = Reader('data/Iris Data.txt').load_train_data()
# encoding y data
encoder = LabelEncoder()
encoder.fit(y_train)
encoder.fit(y_test)
encoded_y_train, encoded_y_test = encoder.transform(y_train), encoder.transform(y_test)
y_train, y_test = np_utils.to_categorical(encoded_y_train), np_utils.to_categorical(encoded_y_test)

# mean normalization for the data
mean1, mean2, mean3, mean4 = np.mean(x_train[:, 0]), np.mean(x_train[:, 1]), np.mean(x_train[:, 2]), np.mean(x_train[:, 3])
min1, max1 = x_train[:, 0].min(), x_train[:, 0].max()
min2, max2 = x_train[:, 1].min(), x_train[:, 1].max()
min3, max3 = x_train[:, 2].min(), x_train[:, 2].max()
min4, max4 = x_train[:, 3].min(), x_train[:, 3].max()
x_train[:, 0], x_train[:, 1], x_train[:, 2], x_train[:, 3] = (x_train[:, 0] - mean1) / (max1 - min1), (x_train[:, 1] - mean2) / (max2 - min2), (x_train[:, 2] - mean3) / (max3 - min3), (x_train[:, 3] - mean4) / (max4 - min4)
x_test[:, 0], x_test[:, 1], x_test[:, 2], x_test[:, 3] = (x_test[:, 0] - mean1) / (max1 - min1), (x_test[:, 1] - mean2) / (max2 - min2), (x_test[:, 2] - mean3) / (max3 - min3), (x_test[:, 3] - mean4) / (max4 - min4)


model = RBF()
model.fit(x_train, y_train, epochs=500, learning_rate=0.005)
predictions = model.predict(X=x_test)
acc = model.eval(pred=predictions, true=y_test)
print(predictions)
print(acc)
plt.xlabel("Number of Epochs")
plt.ylabel("Cost")
plt.plot(model.epochs, model.cost)
plt.show()
#print(x_train)
