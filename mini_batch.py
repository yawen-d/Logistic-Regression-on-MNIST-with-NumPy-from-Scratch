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


# model = {}
# model['W1'] = np.random.randn(num_outputs,num_inputs) / np.sqrt(num_inputs)

# model_grads = copy.deepcopy(model)

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
    w -- weights, a ndarray of size (num_outputs, num_inputs)
    x -- a numpy array of size (num_inputs) -- the MNIST data representing pixels
    y -- true "label" corresponding to x (float)

    Return:
    grad -- gradient of the loss with respect to w, thus same shape as w
    """
    E = np.array([0]*10).reshape((1,10))
    E[0][y] = 1
    A = (E - softmax(np.matmul(w, x))).reshape((10,1))
    grad = np.squeeze(x[0]*A)

    for i in range(1,len(x)):
        grad = np.column_stack((grad,np.squeeze(x[i]*A)))

    return (-1)*grad

def optimize(w, grad, learning_rate):
    return w - learning_rate * grad
    

def model (initial_w, X_train, Y_train, batch_size = 5, num_iterations = 1000, learning_rate = 0.5):
    w = initial_w

    train_size = len(x_train)
    rand_indices = np.random.choice(train_size, (num_iterations,batch_size), replace=True)

    
    for line in rand_indices:
        grad_list = np.zeros((num_outputs,num_inputs))
        for i in range(batch_size):
            grad_list += gradient(w,X_train[line[i]],Y_train[line[i]])

        batch_grad = grad_list/batch_size    
        w = optimize(w,batch_grad,learning_rate)

    return w 

def predict (w, x):
    dist = softmax(np.matmul(w, x))

    return np.argmax(dist)


initial_w = np.random.randn(num_outputs,num_inputs) / np.sqrt(num_inputs)


w = model(initial_w, x_train,y_train, batch_size = 5, num_iterations = 10000, learning_rate = 0.005)




######################################################################################
# test data
total_correct = 0
for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    prediction = predict (w , x)
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test) ) )

