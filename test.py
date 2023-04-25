# %% loading keras LSTM and other libs
from keras.models import Sequential
from keras.layers import Dense, LSTM, RNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %% 
# manual training data
data = pd.read_csv('datasets/graduation_rate.csv')
data1 = np.array(data['ACT composite score'])
data2 = np.array(data['SAT total score'])
data3 = np.array(data['parental income'])
data4 = np.array(data['high school gpa'])
data5 = np.array(data['college gpa'])
# %%
def segmentation(start,end):
    X = []
    y = []
    for i in range(start,end):
        segment = [[data1[i]],[data2[i]],[data3[i]],[data4[i]]]
        X.append(segment)
        y.append([((data5[i]/4)-0.4)])
    return np.array(X),np.array(y)

# %%
segment_length = len(data)
divide = int(len(data)*0.8)
X_train , y_train = segmentation(0,divide)
X_test , y_test = segmentation(divide,len(data))

print(X_train.shape)
# print(len(X_train))
# print(y_train)
# print(len(y_train))
# X_train = np.array([[[0,1,0,1]],[[0,0,1,1]]])
# print(X_train.shape)
# y_train = np.array([[0],[1]])
# print(y_train.shape)

# %% Creating the neural network model
model = Sequential()

# Adding the layers to the model
model.add(LSTM(4, input_shape=(4,1))) # LSTM layer with 8 neurons, input shape of (1, 4)
model.add(Dense(32, activation='relu')) # Hidden layer with neurons
model.add(Dense(32, activation='relu')) # Hidden layer with neurons
model.add(Dense(64, activation='relu')) # Hidden layer with neurons
model.add(Dense(1, activation='sigmoid')) # Output layer with 1 neuron

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# %% Training the model
history = model.fit(X_train, y_train, epochs=200, batch_size=1)
plt.plot(history.history['loss'], label='Training Loss')
plt.legend()
plt.show()
# %%
# loading test data
# X_test = np.array([[[0,1,0,1]],[[0,0,1,1]]])
# y_test = np.array([[0],[1]])
# evaluating model
score = model.evaluate(X_test, y_test, batch_size=1)
print("Accuracy: %.2f%%" % (score[0]*100))
#%%

# printing pridiction score
y_pred = model.predict(X_test)
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
# %%
pred_data = np.array([[[100],[162500000],[499000000],[4.0]]])
y_pred = model.predict(pred_data)
y_pred
# %%
