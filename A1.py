import numpy as np
import h5py
#data file type h5py
import time
import copy
from random import randint

# cd Desktop/CS\ 398/Assignments/A1/
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))
MNIST_data.close()

####################################################################################
#Implementation of stochastic gradient descent algorithm
#number of inputs
num_inputs = 28*28
#number of outputs
num_outputs = 10



def softmax(z):
    """
    Implement the softmax function

    Arguments:
    z -- a k-length vector (float)

    Return:
    result -- the softmax function evaluated on z, returning a set of probability 
    of length k
    """
    result = 1/sum(np.exp(z)) * np.exp(z)
    assert (result.shape == (len(z),))
    return result

def gradient(w, x, y):
    """
    Implement the gradient 

    Arguments:
    w -- weights, an ndarray of size (num_outputs, num_inputs)
    x -- graphic data, a numpy array of size (num_inputs) -- the MNIST data representing pixels
    y -- true "label" corresponding to x (float)

    Return:
    (-1)*grad -- negative gradient of the loss with respect to w, thus same shape as w
    """
    E = np.array([0]*10).reshape((1,10))
    E[0][y] = 1
    A = (E - softmax(np.matmul(w, x))).reshape((10,1))
    grad = np.squeeze(x[0]*A)

    for i in range(1,len(x)):
        grad = np.column_stack((grad,np.squeeze(x[i]*A)))

    return (-1)*grad
    

def initialize(num_outputs,num_inputs):
    """
    Initialize random weights w
    
    Return: 
    w -- weights, an ndarray of size (num_outputs, num_inputs)
    """
    return np.random.randn(num_outputs,num_inputs) / np.sqrt(num_inputs)

def model (X_train, Y_train, num_iterations = 1000, learning_rate = 0.5):
    """
    Implement the stochastic gradient descent model 

    Arguments:
    X_train -- x_train data set, a 2darray of float in shape (num_training_data, num_inputs)
    Y_train -- y_train data set, a vector of float in shape (num_training_data,)
    num_iterations -- number of iterations to have
    learning_rate -- size of base learning rate

    Return:
    w -- the weights after optimization, an 2darray of size (num_outputs, num_inputs)
    """
    # initialize the random weights
    w = initialize(num_outputs,num_inputs)
    # generate a random list of indices for the training set
    train_size = len(x_train)
    rand_indices = np.random.choice(train_size, num_iterations, replace=True)

    def l_rate(base_rate, ite, num_iterations, schedule = False):
        if schedule == True:
            return base_rate * 10 ** (-np.floor(ite/num_iterations*4))
        else:
            return base_rate

    for i in rand_indices:
        w = w - gradient(w,X_train[i],Y_train[i]) * l_rate(learning_rate, i, num_iterations)
    return w 

def predict (w, x):
    """
    Predict y based on fitted weights and x

    Arguments:
    w -- the weights after optimization, an 2darray of size (num_outputs, num_inputs)
    x -- graphic data for testing, a numpy array of size (num_inputs) -- the MNIST data representing pixels
    Return:
    result -- predicted "label" corresponding to x (float)
    """
    dist = softmax(np.matmul(w, x))
    result = np.argmax(dist)
    return result

def testing(w, X_test, Y_test):
    """
    Test the model 

    Arguments:
    w -- the weights after optimization, an 2darray of size (num_outputs, num_inputs)
    X_test -- x_test data set, a 2darray of float in shape (num_testing_data, num_inputs)
    Y_test -- y_test data set, a vector of float in shape (num_testing_data,)
    """
    total_correct = 0
    for n in range(len(X_test)):
        y = Y_test[n]
        x = X_test[n][:]
        prediction = predict (w , x)
        if (prediction == y):
            total_correct += 1
    # print('Accuarcy Test: ',total_correct/np.float(len(X_test)))
    return total_correct/np.float(len(X_test))

for l in [0.01,0.02,0.03]:
    for n in [8000,10000,12000]:
        w = model(x_train, y_train, num_iterations = n, learning_rate = l)
        print('Model #', (n,l))
        print('Number of iterations =',n)
        print('Learning Rate =',l)
        print('Accuarcy Test: ',testing(w,x_test,y_test))
        print('################################')

######################################################################################
# test data


