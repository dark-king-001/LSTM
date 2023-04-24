# %% activating conda enviornment
import os
os.system("conda activate tf")
# %% Activating tensorflow
import tensorflow as tf 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# %% loading keras LSTM and other libs
from keras.models import Sequential
from keras.layers import Dense, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %% Creating the neural network model
model = Sequential()

# Adding the layers to the model
model.add(LSTM(8, input_shape=(1, 4))) # LSTM layer with 8 neurons, input shape of (1, 4)
model.add(Dense(10, activation='relu')) # Hidden layer with 6 neurons
model.add(Dense(10, activation='relu')) # Hidden layer with 6 neurons
model.add(Dense(10, activation='relu')) # Hidden layer with 6 neurons
model.add(Dense(10, activation='relu')) # Hidden layer with 6 neurons
model.add(Dense(1, activation='sigmoid')) # Output layer with 1 neuron

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%

# # Generating sample input data
# X_train = np.random.rand(100, 1, 4)
# X_test = np.random.rand(50, 1, 4)

# # Generating sample labels
# y_train = np.random.randint(2, size=(100, 1))
# y_test = np.random.randint(2, size=(50, 1))

df = pd.read_csv("./datasets/graduation_rate.csv")
df.head(5)

#%%
x_a_axis = 'SAT total score'
x_b_axis = 'parental income'
x_c_axis = 'high school gpa'
x_d_axis = 'college gpa'
y_axis = 'years to graduate'
#%%
data = df[[x_a_axis,x_b_axis,x_c_axis,x_d_axis]]
x = np.array(data).reshape((1000,1, 4))
x.shape
y = np.array(df[y_axis])
# y = y[0].reshape((1,))
#%%
X_train = x[:700]
X_test = x[300:]
y_train = y[:700]
y_test = y[300:]
#%%
X_train[0]
# %%
# Training the model
model.fit(X_train, y_train, epochs=100,validation_data=(X_test, y_test), batch_size=32)

#%%
# Evaluating the model
score = model.evaluate(X_test, y_test, batch_size=32)

# Printing the model's accuracy
print("Accuracy: %.2f%%" % (score[1]*100))


# %% Plot outputs
# plt.scatter(X_train[0], y_train[0], color="black")
# plt.scatter(X_test[0], y_test[0], color="blue", linewidth=3)


# plt.xlabel(x_axis)
# plt.ylabel(y_axis)

# plt.show()
# %%
