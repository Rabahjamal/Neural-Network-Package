import numpy as np
from data_reader import Reader
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
from MLP import MLP


x_train, x_test, y_train, y_test = Reader('data/Iris Data.txt').load_train_data()
encoder = LabelEncoder()
encoder.fit(y_train)
encoder.fit(y_test)
encoded_y_train, encoded_y_test = encoder.transform(y_train), encoder.transform(y_test)
y_train, y_test = np_utils.to_categorical(encoded_y_train), np_utils.to_categorical(encoded_y_test)

# Mean normalization for data
mean1, mean2, mean3, mean4 = np.mean(x_train[:, 0]), np.mean(x_train[:, 1]), np.mean(x_train[:, 2]), np.mean(x_train[:, 3])
min1, max1 = x_train[:, 0].min(), x_train[:, 0].max()
min2, max2 = x_train[:, 1].min(), x_train[:, 1].max()
min3, max3 = x_train[:, 2].min(), x_train[:, 2].max()
min4, max4 = x_train[:, 3].min(), x_train[:, 3].max()
x_train[:, 0], x_train[:, 1], x_train[:, 2], x_train[:, 3] = (x_train[:, 0] - mean1) / (max1 - min1), (x_train[:, 1] - mean2) / (max2 - min2), (x_train[:, 2] - mean3) / (max3 - min3), (x_train[:, 3] - mean4) / (max4 - min4)
x_test[:, 0], x_test[:, 1], x_test[:, 2], x_test[:, 3] = (x_test[:, 0] - mean1) / (max1 - min1), (x_test[:, 1] - mean2) / (max2 - min2), (x_test[:, 2] - mean3) / (max3 - min3), (x_test[:, 3] - mean4) / (max4 - min4)

model = MLP()

model.add_layer(units=8, activation='relu', input_units=4)
model.add_layer(units=3, activation='softmax', input_units=8)
model.fit(X=x_train, y=y_train, epochs=500, learning_rate=0.005)

predictions = model.predict(X=x_test)
print(model.eval(predictions,y_test))
print(predictions)

plt.xlabel("Number of Epochs")
plt.ylabel("Cost")
plt.plot(model.epochs, model.cost)
plt.show()
