# Logistic Regression on MNIST with NumPy from Scratch
Implementing Logistic Regression on MNIST dataset from scratch

## Project Description

Implement and train a logistic regression model from scratch in Python on the MNIST dataset (no PyTorch). The logistic regression model should be trained on the Training Set using stochastic gradient descent. It should achieve 90-93% accuracy on the Test Set.

## Highlights

- **Logistic Regression**
- **SGD with momentum**
- **Learning Rate Decaying**

## Theoretical Derivation

We put the pictures of the mathematical equations and symbols in the algorithm here due to GitHub's incompatibility with math expressions. The raw LaTeX expression is included in `./math_raw.md`.

### Model

Consider a logistic regression model for classification where <img src="https://latex.codecogs.com/svg.latex?\mathcal{Y}&space;=&space;\{0,1,...,K-1\}" title="\mathcal{Y} = \{0,1,...,K-1\}" />, weights <img src="https://latex.codecogs.com/svg.latex?\theta\in\mathbb{R}^{K\times&space;d}" title="\theta\in\mathbb{R}^{K\times d}" /> and biases <img src="https://latex.codecogs.com/svg.latex?b\in&space;\mathbb{R}^K" title="b\in \mathbb{R}^K" />. Given an input <img src="https://latex.codecogs.com/svg.latex?x&space;\in&space;\mathbb{R}^d" title="x \in \mathbb{R}^d" />, the model <img src="https://latex.codecogs.com/svg.latex?f(x;\theta)" title="f(x;\theta)" /> produces a probability of each possible outcome in <img src="https://latex.codecogs.com/svg.latex?\mathcal{Y}" title="\mathcal{Y}" />:

<img src="https://latex.codecogs.com/svg.latex?f(x;\theta,b)=F_{\text{softmax}}(\theta&space;x&plus;b),&space;\\F_{\text{softmax}}(z)=\frac{1}{\sum_{k=0}^{K-1}e^{z_k}}(e^{z_0},e^{z_1},...e^{z_{K-1}})" title="f(x;\theta,b)=F_{\text{softmax}}(\theta x+b), \\F_{\text{softmax}}(z)=\frac{1}{\sum_{k=0}^{K-1}e^{z_k}}(e^{z_0},e^{z_1},...e^{z_{K-1}})" />

where <img src="https://latex.codecogs.com/svg.latex?z_k" title="z_k" /> is the <img src="https://latex.codecogs.com/svg.latex?k" title="k" />-th element of the vector $z$. The set of probabilities on <img src="https://latex.codecogs.com/svg.latex?\mathcal{Y}" title="\mathcal{Y}" /> is <img src="https://latex.codecogs.com/svg.latex?P(\mathcal{Y})&space;:=&space;\{p&space;\in&space;\mathbb{R}^K&space;:\sum^{K-1}_{k=0}&space;p_k&space;=1,p_k&space;\ge&space;0\&space;\forall\&space;k=0,...,K-1\}" title="P(\mathcal{Y}) := \{p \in \mathbb{R}^K :\sum^{K-1}_{k=0} p_k =1,p_k \ge 0\ \forall\ k=0,...,K-1\}" /> The function <img src="https://latex.codecogs.com/svg.latex?F_{\text{softmax}}(z):\mathbb{R}^K\rightarrow&space;\mathcal{P(Y)}" title="F_{\text{softmax}}(z):\mathbb{R}^K\rightarrow \mathcal{P(Y)}" /> is called the “softmax function” and is frequently used in deep learning. <img src="https://latex.codecogs.com/svg.latex?F_{\text{softmax}}(z)" title="F_{\text{softmax}}(z)" /> takes a K-dimensional input and produces a probability distribution on $\mathcal{Y}$. That is, the output of <img src="https://latex.codecogs.com/svg.latex?F_{\text{softmax}}(z)" title="F_{\text{softmax}}(z)" />is a vector of probabilities for the events <img src="https://latex.codecogs.com/svg.latex?0,&space;1,&space;.&space;.&space;.&space;,&space;K&space;-&space;1" title="0, 1, . . . , K - 1" />. The softmax function can be thought of as a smooth approximation to the argmax function since it pushes its smallest inputs towards 0 and its largest input towards 1. 

### Objective Function

The objective function is the negative log-likelihood (commonly referred to in machine learning as the “cross-entropy error”): 

<img src="https://latex.codecogs.com/svg.latex?\mathcal{L}(\theta,b)=\mathbb{E}_{(X,Y)}[\rho(f(X;\theta,b),Y)]," title="\mathcal{L}(\theta,b)=\mathbb{E}_{(X,Y)}[\rho(f(X;\theta,b),Y)]," />

<img src="https://latex.codecogs.com/svg.latex?\rho(z,y)=-\sum_{k=0}^{K-1}\mathbf{1}_{y=k}\&space;\text{log}z_k," title="\rho(z,y)=-\sum_{k=0}^{K-1}\mathbf{1}_{y=k}\ \text{log}z_k," />

<img src="https://latex.codecogs.com/svg.latex?\text{where&space;}z_k&space;\text{&space;is&space;the&space;k-th&space;element&space;of&space;the&space;vector&space;}z\text{&space;and&space;}\mathbf{1}_{y=k}\text{&space;is&space;the&space;indicator&space;function}" title="\text{where }z_k \text{ is the k-th element of the vector }z\text{ and }\mathbf{1}_{y=k}\text{ is the indicator function}" />

<img src="https://latex.codecogs.com/svg.latex?\mathbf{1}_{y=k}=\left\{&space;\begin{array}{ll}&space;1&space;&&space;y=k\\&space;0&space;&&space;y\ne&space;k&space;\end{array}&space;\right." title="\mathbf{1}_{y=k}=\left\{ \begin{array}{ll} 1 & y=k\\ 0 & y\ne k \end{array} \right." />

### Mini-batch Gradient descent

<img src="https://latex.codecogs.com/svg.latex?\theta^{(l&plus;1)}&space;=&space;\theta^{(l)}&space;-\alpha^{(l)}\frac{1}{M}\sum_{m=1}^M&space;\nabla_\theta\rho(f(x^{(l,m)};\theta^{(l)},b^{(l)}),y^{l,m})," title="\theta^{(l+1)} = \theta^{(l)} -\alpha^{(l)}\frac{1}{M}\sum_{m=1}^M \nabla_\theta\rho(f(x^{(l,m)};\theta^{(l)},b^{(l)}),y^{l,m})," />

<img src="https://latex.codecogs.com/svg.latex?b^{(l&plus;1)}&space;=&space;b^{(l)}&space;-\alpha^{(l)}\frac{1}{M}\sum_{m=1}^M&space;\nabla_b\rho(f(x^{(l,m)};\theta^{(l)},b^{(l)}),y^{l,m})" title="b^{(l+1)} = b^{(l)} -\alpha^{(l)}\frac{1}{M}\sum_{m=1}^M \nabla_b\rho(f(x^{(l,m)};\theta^{(l)},b^{(l)}),y^{l,m})" />

### Derivation of SGD

We will derive the stochastic gradient descent algorithm for the logistic regression model. The logistic regression model <img src="https://latex.codecogs.com/svg.latex?f(x;\theta)" title="f(x;\theta)" /> is estimated from the dataset <img src="https://latex.codecogs.com/svg.latex?(x_n,&space;y_n)^N_{n=1}" title="(x_n, y_n)^N_{n=1}" /> where <img src="https://latex.codecogs.com/svg.latex?(x_n,&space;y_n)&space;\sim&space;\mathbb{P}_{X,Y}" title="(x_n, y_n) \sim \mathbb{P}_{X,Y}" />.

The gradient of the loss function for a generic data sample <img src="https://latex.codecogs.com/svg.latex?(x,&space;y)" title="(x, y)" /> is

<img src="https://latex.codecogs.com/svg.latex?\nabla_\theta&space;\rho(f&space;(x;\theta,b),&space;y)&space;=&space;-\nabla_\theta&space;\text{log}F_{\text{softmax},y}&space;(\theta&space;x&plus;b)" title="\nabla_\theta \rho(f (x;\theta,b), y) = -\nabla_\theta \text{log}F_{\text{softmax},y} (\theta x+b)" />

where <img src="https://latex.codecogs.com/svg.latex?F_{softmax,k}(z)" title="F_{softmax,k}(z)" /> is the k-th element of the vector output of the function <img src="https://latex.codecogs.com/svg.latex?F_{softmax,k}(z)" title="F_{softmax,k}(z)" />. Let <img src="https://latex.codecogs.com/svg.latex?\theta_{k,:}" title="\theta_{k,:}" /> be the k-th row of the matrix <img src="https://latex.codecogs.com/svg.latex?\theta" title="\theta" />. 

If <img src="https://latex.codecogs.com/svg.latex?k\ne&space;y" title="k\ne y" />, 

![1](assets/readme_1.png)

If <img src="https://latex.codecogs.com/svg.latex?k=y" title="k=y" />,

![2](assets/readm2_2.png)


Hence, for any k,		

<img src="https://latex.codecogs.com/svg.latex?\nabla_{\theta_{k,:}}\rho(f&space;(x;\theta,b),&space;y)&space;=&space;-(\mathbf{1}_{y=k}-F_{\text{softmax},y}(\theta&space;x&plus;b))x" title="\nabla_{\theta_{k,:}}\rho(f (x;\theta,b), y) = -(\mathbf{1}_{y=k}-F_{\text{softmax},y}(\theta x+b))x" />

Therefore,

<img src="https://latex.codecogs.com/svg.latex?\nabla_{\theta}\rho(f&space;(x;\theta,b),&space;y)&space;=&space;-(e(y)-F_{\text{softmax}}(\theta&space;x&plus;b))x^\top" title="\nabla_{\theta}\rho(f (x;\theta,b), y) = -(e(y)-F_{\text{softmax}}(\theta x+b))x^\top" />

where,

<img src="https://latex.codecogs.com/svg.latex?e(y)=(\mathbf{1}_{y=0},...,\mathbf{1}_{y=K-1})^\top." title="e(y)=(\mathbf{1}_{y=0},...,\mathbf{1}_{y=K-1})^\top." />

Similarly, we could get the gradient of the loss function with respect to <img src="https://latex.codecogs.com/svg.latex?b" title="b" />:

<img src="https://latex.codecogs.com/svg.latex?\nabla_{b}\rho(f(x;\theta,b),&space;y)&space;=&space;-(e(y)-F_{\text{softmax}}(\theta&space;x&plus;b))" title="\nabla_{b}\rho(f(x;\theta,b), y) = -(e(y)-F_{\text{softmax}}(\theta x+b))" />

The stochastic gradient descent algorithm is:

- Randomly initialize the parameter <img src="https://latex.codecogs.com/svg.latex?\theta^{(0)},&space;b^{(0)}" title="\theta^{(0)}, b^{(0)}" />.

- For <img src="https://latex.codecogs.com/svg.latex?l=0,1,...,L" title="l=0,1,...,L" /> :

  - Select <img src="https://latex.codecogs.com/svg.latex?M" title="M" /> data samples <img src="https://latex.codecogs.com/svg.latex?(x^{(l,m)},&space;y^{(l,m)})^M_{m=1}" title="(x^{(l,m)}, y^{(l,m)})^M_{m=1}" /> at random from the dataset <img src="https://latex.codecogs.com/svg.latex?(x_n,&space;y_n)^N_{n=1}" title="(x_n, y_n)^N_{n=1}" />, where <img src="https://latex.codecogs.com/svg.latex?M&space;\ll&space;N" title="M \ll N" />.
  - Calculate the gradient for the loss from the data sample:

  <img src="https://latex.codecogs.com/svg.latex?G_\theta^{(l)}&space;=&space;-\frac{1}{M}\sum_{m=1}^M&space;(e(y^{(l,m)})-F_{\text{softmax}}(\theta^{(l)}&space;x^{(l,m)}&plus;b^{(l)}))(x^{(l,m)})^\top" title="G_\theta^{(l)} = -\frac{1}{M}\sum_{m=1}^M (e(y^{(l,m)})-F_{\text{softmax}}(\theta^{(l)} x^{(l,m)}+b^{(l)}))(x^{(l,m)})^\top" />
  
  <img src="https://latex.codecogs.com/svg.latex?G_b^{(l)}&space;=&space;-\frac{1}{M}\sum_{m=1}^M&space;(e(y^{(l,m)})-F_{\text{softmax}}(\theta^{(l)}&space;x^{(l,m)}&plus;b^{(l)}))" title="G_b^{(l)} = -\frac{1}{M}\sum_{m=1}^M (e(y^{(l,m)})-F_{\text{softmax}}(\theta^{(l)} x^{(l,m)}+b^{(l)}))" />

  - Update the parameters:

  <img src="https://latex.codecogs.com/svg.latex?\theta^{(l&plus;1)}=\theta^{(l)}-\alpha^{(l)}G_\theta^{(l)}," title="\theta^{(l+1)}=\theta^{(l)}-\alpha^{(l)}G_\theta^{(l)}," />
  
  <img src="https://latex.codecogs.com/svg.latex?b^{(l&plus;1)}=b^{(l)}-\alpha^{(l)}G_b^{(l)}" title="b^{(l+1)}=b^{(l)}-\alpha^{(l)}G_b^{(l)}" />
  
where <img src="https://latex.codecogs.com/svg.latex?\alpha^{(l)}" title="\alpha^{(l)}" /> is the learning rate.

## Implementation

The model and main function is included in the script file `logistic_regression.py`. The outputs text are stored in `./logs/` as `.log` files, and the plots for the loss trend and accuracy trend are stored in `./assets/`. All the hyperparameters are stored in `./configs/` as `.json` files. Example: `config_sample.json`

```json
{
    "num_epoches" : 15,
    "batch_size" : 10,
    "learning_rate" : 0.0025,
    "learning_decay" : 0,
    "decay_factor" : 0.75,
    "momentum" : 0,
    "mu" : 0.9
}
```

Note that we implemented available options of **learning decay** and **SGD with momentum**.

In the code, I first loaded the MNIST data, and then set the random seed. After initializing the parameters, I trained the model using mini-batch stochastic gradient descent. If needed, **learning decay** (decay the learning rate by the decay factor when the test accuracy declines or increases by less than 0.1%) and **SGD with momentum**.

**SGD with Momentum:**

```python
if bool(hyp['momentum']) == True:
    w_velocity = mu * w_velocity + learning_rate * dw
    b_velocity = mu * b_velocity + learning_rate * db
    param['w'] -= w_velocity
    param['b'] -= b_velocity
```

**Learning Decay:**

```python
if bool(hyp['learning_decay']) == True:
    try:
        if test_accu_list[-1] - test_accu_list[-2] < 0.001:
            learning_rate *= hyp['decay_factor']
    except:
        pass
```

For each epoch, we evaluate the loss and accuracy.

## Training Results

The training results of 4 configurations are shown as follows. 



|                       |             config_sample              |             config_1              | config_2                          | config_3                          |
| :-------------------: | :------------------------------------: | :-------------------------------: | :---------------------------------: | :---------------------------------: |
|       Optimizer       |                  SGD                   |                SGD                | SGD                               | SGD                               |
|   Number of Epoches   |                   15                   |                15                 | 15                                | 15                                |
|      Batch Size       |                   10                   |              **100**              | **100**                           | 10                                |
|     Learning Rate     |                 0.005                  |            **0.0025**             | **0.0025**                        | 0.005                             |
|    Learning Decay     |                  ----                  |               ----                | ----                              | **0.75**                          |
| Momentum (<img src="https://latex.codecogs.com/svg.latex?\mu" title="\mu" />) |                  ----                  |               ----                | **0.9**                           | ----                              |
|   Optimal Test Loss   |           3310.17 @ Epoch 6            |        4618.44 @ Epoch 15         | 3296.48 @ Epoch 15                | 3310.17 @ Epoch 6                 |
| Optimal Test Accuracy |            0.9093 @ Epoch 6            |         0.8861 @ Epoch 15         | 0.9091 @ Epoch 15                 | 0.9093 @ Epoch 6                  |
|      Loss Trend       | ![loss_sample](assets/loss_sample.png) | ![loss_sample](assets/loss_1.png) | ![loss_sample](assets/loss_2.png) | ![loss_sample](assets/loss_3.png) |
|    Accuracy Trend     | ![loss_sample](assets/accr_sample.png) | ![loss_sample](assets/accr_1.png) | ![loss_sample](assets/accr_2.png) | ![loss_sample](assets/accr_3.png) |

Comments: 

- **Config_sample:** The accuracy drops after epoch 6 . 
- **Config_1:** By increasing the batch size and decreasing the learning rate, the convergence rate decreases because of fewer descent iterations.
- **Config_2:** Adding momentum allows the model to converge faster.
- **Config_3:** Allowing learning decay prevents the accuracy from dropping dramatically.