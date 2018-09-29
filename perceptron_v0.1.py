import copy
import math
import numpy as np

#Global variables

def perceptron_test (X, Y, W, bias):
  total_error = 0
  rows = X.shape[0]
  for i in range (0, rows):
    if ((np.dot(X[i], W) + bias)*Y[i,0]) <= 0:
      total_error += 1
  return total_error

def perceptron(X, Y, rate, bias, epochs):
    cols = X.shape[1]
    #w = np.zeros(cols)
    w = np.random.uniform(-0.02,0.02,cols)
    best_w = np.zeros(cols)
    rows = X.shape[0]

    total_error = 0
    lowest_error = 500
    lowest_error_epoch = 0
    for t in range(epochs):
        total_error = 0
        for i in range (0, rows):
            if ((np.dot(X[i], w) + bias)*Y[i,0]) <= 0:
                w = w + rate*X[i]*Y[i,0]
                bias = bias + (Y[i,0]*rate)
                total_error += 1

        #print ("Epoch ", t, "Total Error", total_error)
        if (total_error < lowest_error):
          lowest_error = total_error
          lowest_error_epoch = t
          best_w = copy.deepcopy(w)
    print ("Lowest error : ",lowest_error, "on epoch", lowest_error_epoch)
    return best_w

def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

def data_in_x_y_format (filename, no_of_columns):
  no_of_rows = file_len (filename)
  data = np.zeros ((no_of_rows, no_of_columns))

  row_no = 0
  with open(filename) as f:
    for line in f:
      words = line.split()
      start = 1
      for word in words:
        if  start == 1:
          start = 0
          data[row_no][0] = int (word)
        else:
          parts = word.split(':')
          column = int (parts[0])
          value = float (parts[1])
          data[row_no][column] = value
      row_no += 1

  raw_Y = copy.deepcopy(data)
  Y = np.delete (raw_Y, np.s_[1:no_of_columns], axis=1)
  X = np.delete (data, 0, axis=1)
  return X, Y

# Input parameters
#train_filename, test_filename, learning_rate, bias, epochs = input().split()
train_filename = "diabetes.train"
test_filename  = "diabetes.test"
learning_rate  = 0.01
bias           = 0.01
epochs         = 20 
no_of_columns  = 20

#Core W learning
X, Y = data_in_x_y_format (train_filename, no_of_columns)
W = perceptron(X,Y,learning_rate,bias,epochs)

# Testing segment
X, Y = data_in_x_y_format (test_filename, no_of_columns)
errors = perceptron_test (X, Y, W, bias)

# Accuracy
accuracy = errors/(X.shape[0])
print ("Test Accuracy :", (1-accuracy)*100, "%")
