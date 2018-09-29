import numpy as np

X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],

])

y = np.array([-1,-1,1,1,1])

def perceptron_sgd(X, Y):
    w = np.zeros(len(X[0]))
    eta = 0.5
    epochs = 20
    total_error = 0

    for t in range(epochs):
        total_error = 0
        for i, x in enumerate(X):
            if (np.dot(X[i], w)*Y[i]) <= 0:
                print ("Made a mistake")
                print ("       Epoch ", t, "i=", i, "X[i]", X[i], "Y[i]", Y[i], "Old W", w)
                w = w + eta*X[i]*Y[i]
                print ("       New W = Old W + X[i] * Y[i] = ", w)
                total_error += 1
            else:
                print ("Yay right prediction")
                print ("       Epoch ", t, "i=", i, "X[i]", X[i], "Y[i]", Y[i], "New W", w)

        print ("Epoch ", t, "Total Error", total_error)

    return w

w = perceptron_sgd(X,y)
print(w)
