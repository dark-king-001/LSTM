# %% activating conda enviornment
import os
os.system("conda activate tf")
# %% Activating tensorflow
import tensorflow as tf 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# %% importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %% importing additional module
from statsmodels.tsa.seasonal import seasonal_decompose

file = open('model.txt','rb')
# %% loading sklearn
from sklearn.preprocessing import MinMaxScaler
# %% loading kera timeseries generator
from keras.preprocessing.sequence import TimeseriesGenerator

# %% loading keras LSTM and other libs
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# %% Load the pickled model
import pickle
with open('model', 'rb') as file:
    response = pickle.load(file)
# %%
