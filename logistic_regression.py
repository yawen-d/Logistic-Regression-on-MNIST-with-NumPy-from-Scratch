import numpy as np
import h5py
#data file type h5py
import time
import copy
from random import randint

# load MNIST data
def load_mnist(filename):
    """
    load the mnist data

    Arguments:
    filename -- the filename string

    Return:
    x_train(train_data),y_train(train_label),x_test(test_data),y_test(test_label)
    """
    MNIST_data = h5py.File(filename, 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
    x_test = np.float32(MNIST_data['x_test'][:])
    y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))
    MNIST_data.close()
    return x_train,y_train,x_test,y_test

def softmax(z):
    """
    Implement the softmax function

    Arguments:
    z -- a k-length vector (float)

    Return:
    result -- the softmax function evaluated on z, returning a set of probability 
    of length k
    """
    exp_list = np.exp(z)
    result = 1/sum(exp_list) * exp_list
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
    print(w.shape)
    print(x.shape)
    A = (E - softmax(np.matmul(x, w))).reshape((10,1))
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

def model (x_train, y_train, num_iterations = 1000, learning_rate = 0.5):
    """
    Implement the logistic regression model with stochastic gradient descent 

    Arguments:
    x_train -- x_train data set, a 2darray of float in shape (num_training_data, num_inputs)
    y_train -- y_train data set, a vector of float in shape (num_training_data,)
    num_iterations -- number of iterations to have
    learning_rate -- size of base learning rate

    Return:
    w -- the weights after optimization, an 2darray of size (num_outputs, num_inputs)
    """
    # initialize the random weights
    num_inputs = x_train.shape[1]
    num_class = len(set(y_train))
    w = initialize(num_inputs,num_class)

    # generate a random list of indices for the training set
    train_data_size = x_train.shape[0]
    rand_indices = np.random.choice(train_data_size, num_iterations, replace=True)

    def l_rate(base_rate, ite, num_iterations, schedule = False):
        if schedule == True:
            return base_rate * 10 ** (-np.floor(ite/num_iterations*4))
        else:
            return base_rate

    for i in rand_indices:
        w = w - gradient(w,x_train[i],y_train[i]) * l_rate(learning_rate, i, num_iterations)
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

def testing(w, x_test, y_test):
    """
    Test the model 

    Arguments:
    w -- the weights after optimization, an 2darray of size (num_outputs, num_inputs)
    x_test -- x_test data set, a 2darray of float in shape (num_testing_data, num_inputs)
    y_test -- y_test data set, a vector of float in shape (num_testing_data,)
    """
    total_correct = 0
    for n in range(len(x_test)):
        y = y_test[n]
        x = x_test[n][:]
        prediction = predict (w , x)
        if (prediction == y):
            total_correct += 1
    # print('Accuarcy Test: ',total_correct/np.float(len(X_test)))
    return total_correct/np.float(len(x_test))


def main():
    x_train,y_train,x_test,y_test = load_mnist('MNISTdata.hdf5')

    for l in [0.02]:
        for n in [10000]:
            w = model(x_train, y_train, num_iterations = n, learning_rate = l)
            print('Model #', (n,l))
            print('Number of iterations =',n)
            print('Learning Rate =',l)
            print('Accuarcy Test: ',testing(w,x_test,y_test))
            print('################################')


if __name__ == "__main__":
    main()
