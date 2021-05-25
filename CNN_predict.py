import keras
import numpy as np
import pandas as pd
import pickle
import pickle as pkl
import tensorflow
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy import interp
from keras.models import Sequential
from keras.models import Model
from keras.models import model_from_yaml
from keras.layers import Input,Dense, Activation,Conv2D,Convolution2D,BatchNormalization,Flatten

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn import metrics

np.random.seed(5)
tensorflow.random.set_seed(12)

# Load train_dataset
load_data = np.load('Train_dataset.npy')
train_x_data = load_data[:,:92].reshape(-1,1,23,4)
train_y_data = load_data[:,-1]

# Build model
model = Sequential()
model.add(Conv2D(10, (1, 1), padding='same', activation='relu'))
model.add(Conv2D(10, (1, 2), padding='same', activation='relu'))
model.add(Conv2D(10, (1, 3), padding='same', activation='relu'))
model.add(Conv2D(10, (1, 5), padding='same', activation='relu'))

model.add(BatchNormalization())
model.add(keras.layers.MaxPool2D(pool_size=(1,5), strides=None, padding='valid'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(23, activation='relu'))
model.add(keras.layers.Dropout(rate=0.15)) 
model.add(Dense(1, name='main_output',activation='sigmoid'))

adam = keras.optimizers.Adam(lr=0.00005)
model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])

# Train data
history = model.fit(train_x_data,train_y_data,batch_size=550, epochs=30, shuffle=True)
loss,accuracy = model.evaluate(train_x_data,train_y_data)

print(loss)
print(accuracy)

# Predict data
Data_test = pd.read_excel('Test_dataset.xlsx')
code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
test_off_target = Data_test['off_target']
test_target = Data_test['WT_target']
test_data = []

for i in range(len(test_target)):
    test_off_seq = test_off_target[i]
    test_target_seq = test_target[i]
    test_off_encoding = [code_dict[i] for i in test_off_seq]
    test_target_encoding = [code_dict[i] for i in test_target_seq]
    test_combined = np.bitwise_or(test_target_encoding,test_off_encoding)
    test_data.append(test_combined)

test_data_x = np.array(test_data,dtype='float64').reshape(-1,1,23,4)
test_y_pred = model.predict(test_data_x)