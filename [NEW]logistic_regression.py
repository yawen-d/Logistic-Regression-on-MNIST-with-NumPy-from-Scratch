import numpy as np
import h5py
#data file type h5py
import time
import copy
import matplotlib.pyplot as plt

# load MNIST data
def load_mnist(filename):
    MNIST_data = h5py.File(filename, 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
    x_test = np.float32(MNIST_data['x_test'][:])
    y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))
    MNIST_data.close()
    return x_train,y_train,x_test,y_test

def initialize(num_inputs,num_classes):
    # num_inputs = 28*28 = 784
    # num_classes = 10
    w = np.random.randn(num_classes, num_inputs) / np.sqrt(num_classes*num_inputs) # (10*784)
    b = np.random.randn(num_classes, 1) / np.sqrt(num_classes) # (10*1) 
    
    param = {
        'w' : w, # (10*784)
        'b' : b  # (10*1)
    }
    return param

def softmax(z):
    exp_list = np.exp(z)
    result = 1/sum(exp_list) * exp_list
    result = result.reshape((len(z),1))
    assert (result.shape == (len(z),1))
    return result

def neg_log_loss(pred, label):
    loss = -np.log(pred[int(label)])
    return loss

def mini_batch_gradient(param, x_batch, y_batch):
    batch_size = x_batch.shape[0]
    w_grad_list = []
    b_grad_list = []
    batch_loss = 0
    for i in range(batch_size):
        x,y = x_batch[i],y_batch[i]
        x = x.reshape((784,1)) # x: (784,1)
        E = np.zeros((10,1)) #(10*1)
        E[y][0] = 1 
        pred = softmax(np.matmul(param['w'], x)+param['b']) #(10*1)

        loss = neg_log_loss(pred, y)
        batch_loss += loss

        w_grad = E - pred
        w_grad = - np.matmul(w_grad, x.reshape((1,784)))
        w_grad_list.append(w_grad)

        b_grad = -(E - pred)
        b_grad_list.append(b_grad)

    dw = sum(w_grad_list)/batch_size
    db = sum(b_grad_list)/batch_size
    return dw, db, batch_loss

def eval(param, x_data, y_data):
    # w: (10*784), x: (10000*784), y:(10000,)
    loss_list = []
    w = param['w'].transpose()
    dist = np.array([np.squeeze(softmax(np.matmul(x_data[i], w))) for i in range(len(y_data))])

    result = np.argmax(dist,axis=1)
    accuracy = sum(result == y_data)/float(len(y_data))

    loss_list = [neg_log_loss(dist[i],y_data[i]) for i in range(len(y_data))]
    loss = sum(loss_list)
    return loss, accuracy

def train(param, hyp ,x_train,y_train,x_test,y_test):
    num_epoches = hyp['num_epoches']
    batch_size = hyp['batch_size']
    learning_rate = hyp['learning_rate']
    mu = hyp['mu']
    test_loss_list, test_accu_list = [],[]
    if hyp['momentum'] == True:
        w_velocity = np.zeros(param['w'].shape)
        b_velocity = np.zeros(param['b'].shape) 

    for epoch in range(num_epoches):
        
        # select the random sequence of training set
        rand_indices = np.random.choice(x_train.shape[0],x_train.shape[0],replace=False)
        num_batch = int(x_train.shape[0]/batch_size)
        batch_loss100 = 0
        
        if hyp['learning_schedule'] == True:
            try:
                if test_accu_list[-1] - test_accu_list[-2] < 0.001:
                    learning_rate *= 0.95
            except:
                pass
            print('learning rate:', learning_rate)
        # for each batch of train data
        for batch in range(num_batch):
            index = rand_indices[batch_size*batch:batch_size*(batch+1)]
            x_batch = x_train[index]
            y_batch = y_train[index]

            # calculate the gradient w.r.t w and b
            dw, db, batch_loss = mini_batch_gradient(param, x_batch, y_batch)
            batch_loss100 += batch_loss
            # update the parameters with the learning rate
            if hyp['momentum'] == True:
                w_velocity = mu * w_velocity + learning_rate * dw
                b_velocity = mu * b_velocity + learning_rate * db
                param['w'] -= w_velocity
                param['b'] -= b_velocity
            else:
                param['w'] -= learning_rate * dw
                param['b'] -= learning_rate * db
            if batch % 100 == 0:
                print('Epoch %d, Batch %d, Loss %.2f' % (epoch+1, batch, batch_loss))
                batch_loss100 = 0
        train_loss, train_accu = eval(param,x_train,y_train)
        test_loss, test_accu = eval(param,x_test,y_test)
        test_loss_list.append(test_loss)
        test_accu_list.append(test_accu)
        print('Epoch %d, Train Loss %.2f, Train Accu %.4f, Test Loss %.2f, Test Accu %.4f' % (epoch+1, train_loss, train_accu, test_loss, test_accu))
    return test_loss_list, test_accu_list



def plot(loss_list, accu_list):
    # epoch_list = list(range(len(loss_list)))
    plt.plot(loss_list)
    plt.ylabel('Loss Function')
    plt.xlabel('Epoch')
    plt.xticks(rotation=60)
    plt.title('Loss Function ~ Epoch')
    plt.savefig('assets/loss_trend.png')
    plt.show()

    plt.plot(accu_list)
    plt.ylabel('Test Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(rotation=60)
    plt.title('Test Accuracy ~ Epoch')
    plt.savefig('assets/accr_trend.png')
    plt.show()

def main(): 
    hyperpara = {
        'num_epoches' : 50,
        'batch_size' : 64,
        'learning_rate' : 0.02,
        'learning_schedule' : True,
        'momentum' : False,
        'mu' : 0.9
    }

    # loading MNIST data
    x_train,y_train,x_test,y_test = load_mnist('MNISTdata.hdf5')

    # setting the random seed
    np.random.seed(1024)

    # initialize the parameters
    num_inputs = x_train.shape[1]
    num_classes = len(set(y_train))
    param = initialize(num_inputs,num_classes)

    # train the model
    loss_list, accu_list = train(param,hyperpara,x_train,y_train,x_test,y_test)

    # plot the loss and accuracy
    plot(loss_list, accu_list)

if __name__ == "__main__":
    main()