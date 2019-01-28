# MNIST-with-Logistic-Regression-from-Scratch
Implementing Logistic Regression on MNIST dataset from scratch

## Project Description:

Implement and train a logistic regression model from scratch in Python for the MNIST dataset (no PyTorch). The logistic regression model should be trained on the Training Set using stochastic gradient descent. It should achieve 90-93% accuracy on the Test Set.

## Implementation

In my code, I first initialize a random set of parameters, and then I use stochastic logistic regression algorithm to train the model with data replacement. Then I test the data based on the training dataset to get a accuracy scores.

I also tested the model for 100 times with random initialization and plotted the histogram of accuracy scores to see the accuracy of my model.

Note that the number of iterations is 100000, and I implemented a learning rate schedule as follows:

![Alt text](asset/Figure_2.png/?raw=true "Learning Rate Schedule")

I wrote 6 functions including `softmax(z)`, `gradient(w, x, y)`, `initialize(num_outputs,num_inputs)`, `model(X_train, Y_train, num_iterations, learning_rate)`, `predict(w, x)`, `testing(model, X_test, Y_test)` to handle initialization, model fitting and testing.

## Test Accuracy 

The test accuracy are shown as follows. Note that all 100 tests have achieved 90% accuracy and the majority of tests have accuracy scores of around 91%.
![Alt text](asset/Figure_1.png/?raw=true "Accuracy Scores in 100 Test")

