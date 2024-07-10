import numpy as np
import pandas as pd

#for reading csv files and preparing training/testing sets
from pandas import read_csv
from sklearn.model_selection import train_test_split

#for the machine learning model
import tensorflow as tf
import keras
from keras.models import load_model
from keras import metrics
from keras.models import Model
from keras.layers import Input, Dense, Dropout

#function for reading csv files and preparing training/testing sets
def read_data(dataset, ratio):
  '''
  dataset: a string stating the name of the csv file
  ratio: a float stating the ratio for splitting the dataset into training and testin set
  '''
  data = read_csv(dataset).to_numpy()
  #first column in the dataset corresponds to the output variable, remaining columns are the input variables
  x = data[:,1:]
  y = data[:,0]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio)
  return x_train, y_train,  x_test, y_test

#function for the machine learning model
def ffnn(x_train, y_train, x_test, y_test, name):
  '''
  x_train: a variable showing the training set inputs
  y_train: a variable showing the training set output
  x_test: a variable showing the training set inputs
  y_test: a variable showing the training set outputs
  name: a string stating the name of the model to be saved
  '''
  inputs = Input(shape=np.shape(x_train)[1])
  n = Dense(10, activation='relu')(inputs)
  n = Dropout(0.2, input_shape=(10,))(n, training=True)
  n = Dense(40, activation='relu')(n)
  n = Dropout(0.2, input_shape=(40,))(n, training=True)
  n = Dense(60, activation='relu')(n)
  n = Dropout(0.2, input_shape=(60,))(n, training=True)
  n = Dense(40, activation='relu')(n)
  n = Dropout(0.2, input_shape=(40,))(n, training=True)
  n = Dense(10, activation='relu')(n)
  outputs = Dense(1)(n)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(optimizer='adam', loss='mse')
  callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

  model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[callback])
  model.save(name)

  return (model.evaluate(x_train, y_train)), (model.evaluate(x_test, y_test))

  '''
  A fully connected feed forward neural network is used as the machine learning model in this script
  However, UQDIR can be implemented with any type of machine learning model
  Should they prefer using this function we provide here, users may revise the model architecture, parameters, and training hyperparameters according to their dataset
  '''

#function for the uncertainty quantification method
def bootstrap(model, x_train, y_train, T):
  '''
  model: machine learning model for which bootstrapping will be performed
  x_train: a variable showing the training set inputs
  y_train: a variable showing the training set output
  T: an integer stating the number of bootstrapping samples
  '''
  bootstrap = []
  for _ in range(T):
    idx = np.random.choice(np.arange(len(x_train)), int(len(x_train)*0.8), replace=False)
    x_sampled = x_train[idx]
    y_sampled = y_train[idx]
    model.fit(x_sampled, y_sampled)
    bootstrap += [model.predict(x_train)]

    predictive_mean = np.mean(bootstrap, axis=0)
    predictive_variance = np.var(bootstrap, axis=0)

    return predictive_mean, predictive_variance

  '''
  Bootstrapping is used as the uncertainty quantification method in this script
  However, UQDIR can be implemented with any type of uncertainty quantification method
  Should they prefer using this function we provide here, users may revise the model algorithm parameters according to their dataset
  '''

def uqdir (beta, theta, x_train, y_train, x_test, y_test, model_name):
  '''
  beta: a float indicating the tolerance for algorithm stopping criterion 1
  theta: a float indicating the tolerance for algorithm stopping criterion 1
  x_train: a variable showing the training set inputs
  y_train: a variable showing the training set outputs
  x_test: a variable showing the testing set inputs
  y_test: a variable showing the testing set outputs
  model_name: a string stating the name of the model to be saved
  '''
  j=10000000 #a very large integer
  mse_train = []
  mse_test = []
  picp = []
  error = {}

  for k in range(j):

    #fit a model and calculate its training/testing errors
    error_train, error_test = ffnn(x_train, y_train, x_test, y_test, model_name)
    mse_train.append(error_train)
    mse_test.append(error_test)

    #perform uncertainty quantification to get the predictive mean and predictive variance
    model = load_model(model_name)
    predictive_mean, predictive_variance = np.array(bootstrap(model, x_train, y_train, T=50))

    #for all the training samples check if they are contained within their prediction intervals
    error = {}
    for i in range(len(x_train)):
      if not ((y_train[i] >= (predictive_mean[i]-2*np.sqrt(predictive_variance[i]))) and (y_train[i] <= (predictive_mean[i]+2*np.sqrt(predictive_variance[i])))):
        #for the training samples not contained within their prediction intervals calculate the distance between the upper or lower prediction bound and the true output value
        error[i] = abs(abs(y_train[i]-predictive_mean[i])-2*np.sqrt(predictive_variance[i]))
    picp.append((len(x_train)-len(error))/len(x_train))

    '''
    After this point, algorithm stopping criateria is checked
    Criterion 1: The algorithm stops when the predictive accuracy of the model starts to decline due to resampling
    Criterion 2: The algorithm stops if the coverage of the prediction intervals starts to decline due to resampling
    Criterion 3: The algorithm stops when all of the training samples are covered by their respective prediction intervals
    '''
    if (mse_test[-1] > mse_test[0]*beta):
      break
    if len(mse_test) >= 6 and ((mse_test[-1] > mse_test[-2]*beta) or (mse_test[-1] > mse_test[-3]*beta) or (mse_test[-1] > mse_test[-4]*beta) or (mse_test[-1] > mse_test[-5]*beta) or (mse_test[-1] > mse_test[-6]*beta)):
      break
    if (picp[-1] < picp[0]*theta):
      break
    if len(picp) >= 6 and ((picp[-1] < picp[-2]*theta) or (picp[-1] < picp[-3]*theta) or (picp[-1] < picp[-4]*theta) or (picp[-1] < picp[-5]*theta) or (picp[-1] < picp[-6]*theta)):
      break
    if len(error) == 0:
      break

    #selecting the sample from the training set to be resampled and calculating its weight
    point = max(error, key=error.get)
    total_error = sum(error.values())
    count = np.ceil((error[point]/total_error)*len(x_train))

    #adding the selected sample to the training set
    for j in range(int(count)):
      x_train = np.concatenate([x_train, np.array(x_train[point].reshape(1, -1))], axis=0)
      y_train = np.append(y_train, np.array(y_train[point]))

  return mse_test, picp
