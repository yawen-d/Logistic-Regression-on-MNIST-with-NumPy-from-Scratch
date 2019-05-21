import numpy as np
import h5py
# data file type h5py
import time
import copy
import matplotlib.pyplot as plt
from cfg import loadConfig 


def load_mnist(filename):
    """load MNIST data"""
    MNIST_data = h5py.File(filename, 'r')
    x_train = np.float32(MNIST_data['x_train'][:])
    y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
    x_test = np.float32(MNIST_data['x_test'][:])
    y_test = np.int32(np.array(MNIST_data['y_test'][:,0]))
    MNIST_data.close()
    return x_train,y_train,x_test,y_test

def initialize(num_inputs,num_classes):
    """initialize the parameters"""
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
    """implement the softmax functions
    input: numpy ndarray
    output: numpy ndarray
    """
    exp_list = np.exp(z)
    result = 1/sum(exp_list) * exp_list
    result = result.reshape((len(z),1))
    assert (result.shape == (len(z),1))
    return result

def neg_log_loss(pred, label):
    """implement the negative log loss"""
    loss = -np.log(pred[int(label)])
    return loss

def mini_batch_gradient(param, x_batch, y_batch):
    """implement the function to compute the mini batch gradient
    input: param -- parameters dictionary (w, b)
           x_batch -- a batch of x (size, 784)
           y_batch -- a batch of y (size,)
    output: dw, db, batch_loss
    """
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
    """ implement the evaluation function
    input: param -- parameters dictionary (w, b)
           x_data -- x_train or x_test (size, 784)
           y_data -- y_train or y_test (size,)
    output: loss and accuracy
    """
    # w: (10*784), x: (10000*784), y:(10000,)
    loss_list = []
    w = param['w'].transpose()
    dist = np.array([np.squeeze(softmax(np.matmul(x_data[i], w))) for i in range(len(y_data))])

    result = np.argmax(dist,axis=1)
    accuracy = sum(result == y_data)/float(len(y_data))

    loss_list = [neg_log_loss(dist[i],y_data[i]) for i in range(len(y_data))]
    loss = sum(loss_list)
    return loss, accuracy

def train(param, hyp , x_train, y_train, x_test, y_test,cfg_idx):
    """ implement the train function
    input: param -- parameters dictionary (w, b)
           hyp -- hyperparameters dictionary
           x_train -- (60000, 784)
           y_train -- (60000,)
           x_test -- x_test (10000, 784)
           y_test -- y_test (10000,)
    output: test_loss_list, test_accu_list
    """
    num_epoches = hyp['num_epoches']
    batch_size = hyp['batch_size']
    learning_rate = hyp['learning_rate']
    mu = hyp['mu']
    test_loss_list, test_accu_list = [],[]
    if bool(hyp['momentum']) == True:
        w_velocity = np.zeros(param['w'].shape)
        b_velocity = np.zeros(param['b'].shape) 

    for epoch in range(num_epoches):
        
        # select the random sequence of training set
        rand_indices = np.random.choice(x_train.shape[0],x_train.shape[0],replace=False)
        num_batch = int(x_train.shape[0]/batch_size)
        batch_loss100 = 0
        
        if bool(hyp['learning_decay']) == True:
            try:
                if test_accu_list[-1] - test_accu_list[-2] < 0.001:
                    learning_rate *= hyp['decay_factor']
            except:
                pass
            
            message = 'learning rate: %.8f' % learning_rate
            print(message)
            logging.info(message)

        # for each batch of train data
        for batch in range(num_batch):
            index = rand_indices[batch_size*batch:batch_size*(batch+1)]
            x_batch = x_train[index]
            y_batch = y_train[index]

            # calculate the gradient w.r.t w and b
            dw, db, batch_loss = mini_batch_gradient(param, x_batch, y_batch)
            batch_loss100 += batch_loss
            # update the parameters with the learning rate
            if bool(hyp['momentum']) == True:
                w_velocity = mu * w_velocity + learning_rate * dw
                b_velocity = mu * b_velocity + learning_rate * db
                param['w'] -= w_velocity
                param['b'] -= b_velocity
            else:
                param['w'] -= learning_rate * dw
                param['b'] -= learning_rate * db
            if batch % 100 == 0:
                message = 'Epoch %d, Batch %d, Loss %.2f' % (epoch+1, batch, batch_loss)
                print(message)
                # logging.info(message)

                batch_loss100 = 0
        train_loss, train_accu = eval(param,x_train,y_train)
        test_loss, test_accu = eval(param,x_test,y_test)
        test_loss_list.append(test_loss)
        test_accu_list.append(test_accu)

        message = 'Epoch %d, Train Loss %.2f, Train Accu %.4f, Test Loss %.2f, Test Accu %.4f' % (epoch+1, train_loss, train_accu, test_loss, test_accu)
        print(message)
        logging.info(message)
    return test_loss_list, test_accu_list



def plot(loss_list, accu_list, cfg_idx):
    """store the plots"""
    # epoch_list = list(range(len(loss_list)))
    plt.plot(loss_list)
    plt.ylabel('Loss Function')
    plt.xlabel('Epoch')
    plt.xticks(rotation=60)
    plt.title('Loss Function ~ Epoch')
    plt.savefig('assets/loss_{}.png'.format(cfg_idx))
    plt.show()

    plt.plot(accu_list)
    plt.ylabel('Test Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(rotation=60)
    plt.title('Test Accuracy ~ Epoch')
    plt.savefig('assets/accr_{}.png'.format(cfg_idx))
    plt.show()

def main(args): 
    cfg_idx = args.config
    cfg_name = 'config_{}.json'.format(cfg_idx)
    hyperpara = loadConfig(cfg_name)

    # loading MNIST data
    x_train,y_train,x_test,y_test = load_mnist('MNISTdata.hdf5')

    # setting the random seed
    np.random.seed(1024)

    # initialize the parameters
    num_inputs = x_train.shape[1]
    num_classes = len(set(y_train))
    param = initialize(num_inputs,num_classes)

    # train the model
    loss_list, accu_list = train(param,hyperpara,x_train,y_train,x_test,y_test, cfg_idx)

    # plot the loss and accuracy
    plot(loss_list, accu_list, cfg_idx)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, 
                        default="sample", help="Config of hyperparameters")
    args = parser.parse_args()

    import logging
    logging.basicConfig(filename="./logs/{}.log".format(args.config), filemode="w", format="%(message)s", level=logging.DEBUG)
    
    main(args)