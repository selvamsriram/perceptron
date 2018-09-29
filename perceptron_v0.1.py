import copy
import math
import numpy as np

# Peceptron Tester Function
def perceptron_test (X, Y, W, bias):
  total_error = 0
  rows = X.shape[0]
  for i in range (0, rows):
    if ((np.dot(X[i], W) + bias)*Y[i,0]) <= 0:
      total_error += 1
  return total_error



# Perceptron Learner Function
def perceptron(X, Y, W, rate, bias, epochs):
    cols = X.shape[1]
    rows = X.shape[0]

    total_error = 0
    lowest_error = 500
    lowest_error_epoch = 0
    for t in range(epochs):
        total_error = 0
        for i in range (0, rows):
            if ((np.dot(X[i], W) + bias)*Y[i,0]) <= 0:
                W = W + rate*X[i]*Y[i,0]
                bias = bias + (Y[i,0]*rate)
                total_error += 1
    return W, bias



# Utility Function
def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1


# Utility Function
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



# Cross validation function
def cross_validation (kfold, eta, bias, epochs, no_of_columns, W):
  accuracy = 0
  for i in range (0, kfold):
    training_filenames = []
    for j in range (0, kfold):
      if (i != j):
        training_filenames.append ('training0'+str(j)+'.data')

    with open ('temporary.data', 'w') as temp_file:
      for fname  in training_filenames:
        with open(fname) as iterfile:
          for line in iterfile:
            temp_file.write (line)

    #Cross Validation Training 
    X, Y = data_in_x_y_format ('temporary.data', no_of_columns)
    new_W, new_bias = perceptron(X,Y,W,eta,bias,epochs)

    #Cross Validation Testing
    X, Y = data_in_x_y_format ('training0'+str(i)+'.data', no_of_columns)
    errors = perceptron_test (X, Y, new_W, new_bias)

    accuracy += (1-(errors/(X.shape[0])))*100
  return (accuracy/kfold)

eta             = [1, 0.1, 0.01]
kfold           = 5
no_of_columns   = 20
bias            = np.random.uniform(-0.01,0.01)
epochs          = 10 
W               = np.random.uniform(-0.01, 0.01, no_of_columns-1)
best_eta        = 0
best_accuracy   = 0 

print ("Cross validation results ")
print ("===================================================================")
#Basic cross validation (Only ETA)
for current_eta in eta:
  W_copy = copy.deepcopy (W)
  accuracy =  cross_validation (kfold, current_eta, bias, epochs, no_of_columns, W_copy)
  print ("ETA : ", current_eta, "Average Cross Validation Accuracy : ", accuracy, "%")
  if (accuracy > best_accuracy):
    best_accuracy = accuracy
    best_eta = current_eta

best_accuracy = 0
best_epoch = 0
best_w = np.zeros(no_of_columns - 1)
best_bias = 0
#Train the perceptron now
for i in range (1, 21):
  X, Y = data_in_x_y_format ('diabetes.train', no_of_columns)
  new_W, new_bias = perceptron(X, Y, W, best_eta, bias, i)

  X, Y = data_in_x_y_format ('diabetes.dev', no_of_columns)
  errors = perceptron_test (X, Y, new_W, new_bias)
  accuracy = (1-(errors/(X.shape[0])))*100
  if (accuracy > best_accuracy):
    best_accuracy = accuracy
    best_epoch = i
    best_w = copy.deepcopy (new_W)
    best_bias = new_bias

print ("")
print ("Best epoch and eta on development set")
print ("===================================================================")
print ("Best epoch : ", best_epoch, "best eta", best_eta, "accuracy is ", best_accuracy, "%")

X, Y = data_in_x_y_format ('diabetes.test', no_of_columns)
errors = perceptron_test (X, Y, best_w, best_bias)
accuracy = (1-(errors/(X.shape[0])))*100
print ("")
print ("Results on the test file as follows")
print ("===================================================================")
print ("Accuracy : ", accuracy, "%")
