# %% loading keras LSTM and other libs
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# %% Creating the neural network model
model = Sequential()

# Adding the layers to the model
model.add(LSTM(8, input_shape=(1,4))) # LSTM layer with 8 neurons, input shape of (1, 4)
model.add(Dense(10, activation='relu')) # Hidden layer with 6 neurons
model.add(Dense(10, activation='relu')) # Hidden layer with 6 neurons
model.add(Dense(10, activation='relu')) # Hidden layer with 6 neurons
model.add(Dense(10, activation='relu')) # Hidden layer with 6 neurons
model.add(Dense(1, activation='sigmoid')) # Output layer with 1 neuron

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# %% 
# manual training data
X_train = np.array([[[0,1,0,1]],[[0,0,1,1]]])
y_train = np.array([[0],[1]])
# %% Training the model
history = model.fit(X_train, y_train, epochs=1000, batch_size=32)
plt.plot(history.history['loss'], label='Training Loss')
plt.legend()
plt.show()
#%%
# loading test data
X_test = np.array([[[0,1,0,1]],[[0,0,1,1]]])
y_test = np.array([[0],[1]])
# evaluating model
score = model.evaluate(X_test, y_test, batch_size=32)
print("Accuracy: %.2f%%" % (score[1]*100))

# printing pridiction score
y_pred = model.predict(X_test)
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

# %%
