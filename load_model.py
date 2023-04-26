# %% loading keras LSTM and other libs
from keras.models import Sequential
from keras.layers import Dense, LSTM, RNN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#%%
import pickle
def loadmodel():
    with open(input('Enter model name : '), 'rb') as file:
        return pickle.load(file)
model = loadmodel()
# %%
# %%
pred_data = np.array([[[50],[4000],[40],[1.0]]])
y_pred = model.predict(pred_data)*4
y_pred
# %%
