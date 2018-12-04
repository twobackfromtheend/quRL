import logging
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, CuDNNLSTM, SimpleRNN, TimeDistributed
import matplotlib.pyplot as plt
from tensorflow.python.keras.optimizers import Adam

BATCH_SIZE = 1
INPUTS = 1
OUTPUTS = 1

non_stateful_model = keras.Sequential()
input_shape = (1, INPUTS)
non_stateful_model.add(CuDNNLSTM(24, input_shape=input_shape, return_sequences=True))
non_stateful_model.add(TimeDistributed(Dense(OUTPUTS, activation='linear')))
optimizer = 'adam'#Adam(lr=1)
non_stateful_model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])


data_length = 100
def get_data():
    x = np.linspace(0, 50, data_length)
    coeff = -1 if random.getrandbits(1) else 1
    y = coeff * np.sin(x) + (np.random.random(data_length) - 0.5) * 2 * 0.05

    y_1 = np.roll(y, 1)
    y_1[0] = 0
    return x, y, y_1


x, y, y_1 = get_data()

plt.plot(x, y)
plt.show()

# Train stateful_model
losses = []
for _ in range(3000):
    x, y, y_1 = get_data()
    losses.append(non_stateful_model.train_on_batch(y_1.reshape((-1, 1, 1)), y.reshape((-1, 1, 1))))
    print(_)

plt.plot(losses)
plt.show()

x, y, y_1 = get_data()

batch_predicted_y = non_stateful_model.predict(y_1.reshape((-1, 1, 1)))
batch_predicted_y = np.array(batch_predicted_y).reshape((-1))

plt.plot(x, batch_predicted_y, label='predicted')
plt.plot(x, y, label='sine')
plt.legend()
plt.show()

reset_states_predicted_y = []
for i in range(data_length):
    predicted_y = non_stateful_model.predict(y_1[i].reshape((1, 1, 1)))
    reset_states_predicted_y.append(predicted_y)
    print(i)
reset_states_predicted_y = np.array(reset_states_predicted_y).reshape((-1))

plt.plot(x, batch_predicted_y, label='predicted')
plt.plot(x, reset_states_predicted_y, label='reset')
plt.plot(x, y, label='sine')
plt.legend()
plt.show()

test_y = np.linspace(-1, 1, 10000)
test_predicted_y = []
for i in range(len(test_y)):
    predicted_y = non_stateful_model.predict(test_y[i].reshape((1, 1, 1)))
    test_predicted_y.append(predicted_y)
    print(i)
test_predicted_y = np.array(test_predicted_y).reshape((-1))

plt.plot(test_y, test_predicted_y, label='reset')
plt.legend()
plt.show()
