# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:44:51 2021

@author: gaurav
"""

# hyper parameter for deep learning by keras tuner
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from kerastuner.tuners import RandomSearch

df = pd.read_csv(r'C:\Users\gaurav\Desktop\Keras-Tuner-main\Keras-Tuner-main\Real_Combine.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

def model_buliding(hp):
    model = Sequential()
    for i in range(hp.Int('num_layer',2,20)):
        model.add(Dense(units=hp.Int('units',min_value = 32,max_value = 512,step = 32)
                        ,activation='relu'))
        model.add(Dense(1, activation="softmax"))
        model.compile(optimizer= keras.optimizers.Adam(hp.Choice('learning_rate',[10e-2,10e-3,10e-4])),
        loss = 'mean_absolute_error',metrics = ['mean_absolute_error'])
        
        return model
    
tuner = RandomSearch(model_buliding,objective = 'val_mean_absolute_error',
                     max_trials = 5,executions_per_trial= 3,
                     project_name = 'Air_quality_index')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size =0.3,random_state =0)

tuner.search(X_train,y_train,epochs = 5,validation_data=(X_test,y_test))
tuner.results_summary()

    







