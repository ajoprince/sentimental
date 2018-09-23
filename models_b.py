#Import data_b from data_processor
from data_processor import data, mat #importing dataset from base or load pickled dataset 
import numpy as np
from sklearn.pipeline import make_pipeline

X, y = mat(data) # sort correct imports

#Create training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)


from xgboost import XGBClassifier

XGB = XGBClassifier(n_estimators = 10)
XGB.fit(X_train, y_train, verbose = False)
#XGB.fit(X_train, y_train, eval_set = [(X_val, y_val)], early_stopping_rounds = 5, eval_metric = 'mlogloss', verbose = False)

#Create ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


#Build neural network - highest val accuracy 0.6328
NN = Sequential()
NN.add(Dense(input_dim = X.shape[1], activation = 'relu', units = 2000))
NN.add(Dropout(0.5))
NN.add(Dense(units = 1500, activation = 'relu'))
NN.add(Dropout(0.5))
NN.add(Dense(units = 1000, activation = 'relu'))
NN.add(Dropout(0.5))
NN.add(Dense(units = 500, activation = 'relu'))
NN.add(Dropout(0.5))
NN.add(Dense(units = 100, activation = 'relu'))
NN.add(Dropout(0.5))
NN.add(Dense(units = 1, activation = 'softmax'))
NN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Train model using trainining data
NN.fit(X, y, batch_size = 32, epochs = 4, validation_split = 0.2)



'''Save classifier to keras.models
NN.save('nn_classifier.h5')
from keras.models import load_model

NN = load_model('nn_classifier.h5')
'''
