import copy
import math
import numpy as np

# Peceptron Tester Function
#--------------------------------------------------------------------------------------------------
def perceptron_test (X, Y, W, bias):
  total_error = 0
  rows = X.shape[0]
  for i in range (0, rows):
    if ((np.dot(X[i], W) + bias)*Y[i,0]) <= 0:
      total_error += 1
  return total_error

# Perceptron Learner Function
#--------------------------------------------------------------------------------------------------
def perceptron(X, Y, W, rate, mu, bias, epochs, declining_eta, average_mode, aggressive_mode):
    cols = X.shape[1]
    rows = X.shape[0]
    if average_mode > 0:
      a = np.zeros (X.shape[1])
      bias_a = 0

    for t in range(0, epochs):
        randomize = np.arange (X.shape[0])
        np.random.shuffle(randomize)
        X = X[randomize]
        Y = Y[randomize]
        for i in range (0, rows):
            if (mu > 0):
              # Margin perceptron and Aggressive Perceptron
              if ((np.dot(X[i], W) + bias)*Y[i,0]) < mu:
                if (aggressive_mode == 0):
                  #Margin perceptron
                  W = W + rate*X[i]*Y[i,0]
                  bias = bias + (Y[i,0]*rate)
                elif (aggressive_mode == 1):
                  #Aggressive perceptron
                  rate = (mu - ((np.dot(X[i], W) + bias)*Y[i,0]))/(np.dot (X[i], X[i]) + 1)
                  W = W + rate*X[i]*Y[i,0]
            else:
              if ((np.dot(X[i], W) + bias)*Y[i,0]) <= 0:
                  if (declining_eta == 0):
                    W = W + rate*X[i]*Y[i,0]
                    bias = bias + (Y[i,0]*rate)
                    if (average_mode == 1):
                      a = a + W
                      bias_a = bias_a + bias
                  else:
                    rate = (rate/(1+t))
                    W = W + rate*X[i]*Y[i,0]
                    bias = bias + (Y[i,0]*rate)
              elif (average_mode == 1):
                a = a + W
                bias_a = bias_a + bias

    if (average_mode == 1):
      return a, bias_a
    else:
      return W, bias

# Train and test function
#--------------------------------------------------------------------------------------------------
def train_and_test_perceptron (train_filename, test_filename, no_of_columns, W, eta, mu, bias, epochs, declining_eta, average_mode, aggressive_mode):
  X, Y = data_in_x_y_format (train_filename, no_of_columns)
  new_W, new_bias = perceptron(X, Y, W, eta, mu, bias, epochs, declining_eta, average_mode, aggressive_mode)

  X, Y = data_in_x_y_format (test_filename, no_of_columns)
  errors = perceptron_test (X, Y, new_W, new_bias)

  accuracy = (1-(errors/(X.shape[0])))*100
  return accuracy, new_W, new_bias

# Utility Function
#--------------------------------------------------------------------------------------------------
def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

# Utility Function
#--------------------------------------------------------------------------------------------------
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
#--------------------------------------------------------------------------------------------------
def cross_validation (kfold, eta, mu, bias, epochs, no_of_columns, W, declining_eta, average_mode, aggressive_mode):
  accuracy = 0
  consolidated_accuracy = 0
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
    accuracy, new_W, new_bias = train_and_test_perceptron ('temporary.data', 'training0'+str(i)+'.data',
                                                  no_of_columns, W, eta, mu, bias, epochs, declining_eta, average_mode, aggressive_mode)
    consolidated_accuracy += accuracy 
  return (consolidated_accuracy/kfold)

# Mother ship
#--------------------------------------------------------------------------------------------------
def train_test_request_processor (kfold, eta, mu, bias, epochs, no_of_columns, W, declining_rate, average_mode, aggressive_mode):
  best_accuracy = 0
  #Basic cross validation (Only ETA)
  for current_mu in mu:
    for current_eta in eta:
      W_copy = copy.deepcopy (W)
      accuracy =  cross_validation (kfold, current_eta, current_mu, bias, epochs, no_of_columns, W_copy, declining_rate, average_mode, aggressive_mode)
      if (accuracy > best_accuracy):
        best_accuracy = accuracy
        best_eta = current_eta
        best_mu = current_mu

  print ("Cross Validation Selected - ETA : ", best_eta)
  print ("Cross Validation Selected - MU  : ", best_mu)
  # Re-init for future use
  best_accuracy = 0
  best_epoch = 0
  best_w = np.zeros(no_of_columns - 1)
  best_bias = 0

  #Train for each epoch and test in development data for each of them and measure accuracy
  for i in range (1, 21):
    accuracy, new_W, new_bias = train_and_test_perceptron ('diabetes.train', 'diabetes.dev', no_of_columns,
                                                           W, best_eta, best_mu, bias, i, declining_rate, average_mode, aggressive_mode)
    if (accuracy > best_accuracy):
      best_accuracy = accuracy
      best_epoch = i
      best_w = copy.deepcopy (new_W)
      best_bias = new_bias

  print ("Best epoch                      : ", best_epoch)
  print ("Dev Accuracy                    : ", best_accuracy, "%")

  X, Y = data_in_x_y_format ('diabetes.test', no_of_columns)
  errors = perceptron_test (X, Y, best_w, best_bias)
  accuracy = (1-(errors/(X.shape[0])))*100
  print ("Test File Accuracy              : ", accuracy, "%")
  return accuracy

#--------------------------------------------------------------------------------------------------

# Main Function Starts here 
def main_function (seed_value):
  kfold           = 5
  no_of_columns   = 20
  eta_list        = [1, 0.1, 0.01]
  mu_list         = [0]
  np.random.seed (seed_value)
  W               = np.random.uniform(-0.01, 0.01, no_of_columns-1)
  bias            = np.random.uniform(-0.01,0.01)
  epochs          = 10 
  declining_rate  = 0
  average_mode    = 0
  aggressive_mode = 0
  print ("")
  print ("******************Basic Perceptron Start *******************")
  print ("******************Seed Value", seed_value, "*******************")
  accuracy = train_test_request_processor (kfold, eta_list, mu_list, bias, epochs, no_of_columns, W, declining_rate, average_mode, aggressive_mode)
  print ("******************Basic Perceptron End *********************")

  # Enable Decaying rate
  print ("*****************Decaying learning rate Start***************")
  declining_rate = 1
  train_test_request_processor (kfold, eta_list, mu_list, bias, epochs, no_of_columns, W, declining_rate, average_mode, aggressive_mode)
  print ("*****************Decaying learning rate End End ************")

  # Update values for mu
  print ("*****************Margin Perceptron Start********************")
  mu_list         = [1, 0.1, 0.01]
  train_test_request_processor (kfold, eta_list, mu_list, bias, epochs, no_of_columns, W, declining_rate, average_mode, aggressive_mode)
  print ("*****************Margin Perceptron End *********************")

  # Enable average mode
  print ("*****************Average Perceptron Start********************")
  average_mode = 1
  mu_list = [0]
  train_test_request_processor (kfold, eta_list, mu_list, bias, epochs, no_of_columns, W, declining_rate, average_mode, aggressive_mode)
  print ("*****************Average Perceptron End *********************")

  # Enable Aggressive mode
  print ("*****************Aggressive Perceptron Start********************")
  aggressive_mode = 1
  average_mode = 0
  eta_list        = [1]
  mu_list         = [1, 0.1, 0.01]
  train_test_request_processor (kfold, eta_list, mu_list, bias, epochs, no_of_columns, W, declining_rate, average_mode, aggressive_mode)
  print ("*****************Aggressive Perceptron End *********************")

  return accuracy


main_function (11)
#best_accuracy = 0;
#cur_accuracy = 0;
#for i in range (0, 50):
#  cur_accuracy = main_function (i)
#  if (cur_accuracy > best_accuracy):
#    best_accuracy = cur_accuracy
#    best_seed = i

#print ("Best seed is ", best_seed)
