import numpy as np



class MLP:
    def __init__(self):
        self.__layers = []
        self.cost = []
        self.epochs = []

    def __sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def __tanh(self, z):
        return (1.0-np.exp(-z))/(1.0+np.exp(-z))

    def __relu(self, z):
        zeros = np.zeros(shape=z.shape)
        return np.maximum(z, zeros)

    def __softmax(self, z):
        return (np.exp(z.T) / np.sum(np.exp(z))).T

    def __net_value(self, a, w):
        return np.add(np.dot(a, w[1:]), w[0])

    def add_layer(self, units, activation, input_units):
        weights = np.random.rand(input_units+1, units)
        a = np.zeros(shape=(1, units))
        delta = np.zeros(shape=(1, 1))
        self.__layers.append([units, activation, input_units, weights, a, delta])

    def fit(self, X, y, epochs, learning_rate):

        for epoch in range(epochs):
            item = 0
            errors = []
            for sample in X:

                # forward step
                prv_a = sample
                for i in range(len(self.__layers)):
                    w = self.__layers[i][3]
                    z = self.__net_value(prv_a, w)
                    a = 0
                    if self.__layers[i][1] == 'sigmoid':
                        a = self.__sigmoid(z)
                    elif self.__layers[i][1] == 'tanh':
                        a = self.__tanh(z)
                    elif self.__layers[i][1] == 'relu':
                        a = self.__relu(z)
                    elif self.__layers[i][1] == 'softmax':
                        a = self.__softmax(z)
                    self.__layers[i][4] = a
                    prv_a = a

                # backward step
                self.__layers.reverse()

                error = np.mean(np.square(y[item]-self.__layers[0][4]))
                errors.append(error)

                derv = np.multiply(self.__layers[0][4], (1 - self.__layers[0][4]))
                prv_delta = np.multiply(y[item]-self.__layers[0][4], derv)
                prv_w = self.__layers[0][3]
                self.__layers[0][5] = prv_delta
                l = 1
                for i in range(len(self.__layers[1:])):
                    a = self.__layers[l][4]
                    derv = np.multiply(a, (1 - a))
                    delta = np.multiply(np.dot(prv_delta, np.transpose(prv_w[1:])), derv)
                    self.__layers[l][5] = delta
                    prv_delta = delta
                    prv_w = self.__layers[l][3]
                    l = l + 1

                # weights update step
                self.__layers.reverse()
                prv_a = sample
                for i in range(len(self.__layers)):
                    w = self.__layers[i][3]
                    delta = self.__layers[i][5]
                    a0 = np.ones(shape=(1, sample.shape[0]))
                    delta = np.reshape(delta, [1, delta.shape[0]])
                    prv_a = np.reshape(prv_a, [1, prv_a.shape[0]])
                    w[1:] = w[1:] + learning_rate * np.dot(np.transpose(prv_a), delta)
                    w[0] = w[0] + learning_rate * delta
                    self.__layers[i][3] = w
                    prv_a = self.__layers[i][4]

                item = item + 1
            # calculate the MSE error after ith epoch
            errors = np.array(errors)
            mse = 0.5 * (np.mean(errors))
            self.cost.append(mse)
            self.epochs.append(epoch)
            print(mse)





    def predict(self, X):
        #print(self.__layers)
        # forward step
        prv_a = X
        for i in range(len(self.__layers)):
            w = self.__layers[i][3]
            z = self.__net_value(prv_a, w)
            a = 0
            if self.__layers[i][1] == 'sigmoid':
                a = self.__sigmoid(z)
            elif self.__layers[i][1] == 'tanh':
                a = self.__tanh(z)
            elif self.__layers[i][1] == 'relu':
                a = self.__relu(z)
            elif self.__layers[i][1] == 'softmax':
                a = self.__softmax(z)
            #self.__layers[i][4] = a
            prv_a = a

        output = prv_a
        predictions = np.argmax(output, 1)
        return predictions

    def eval(self, pred, true):
        classes = np.argmax(true, 1)
        num_of_corrects = 0
        for i in range(len(classes)):
            if classes[i] == pred[i]:
                num_of_corrects = num_of_corrects + 1
        return (float(num_of_corrects)/float(len(classes))) * 100.0

