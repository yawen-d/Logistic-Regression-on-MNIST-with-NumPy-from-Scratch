# Logistic Regression on MNIST with NumPy from Scratch
## Theoretical Derivation

### Model

Consider a logistic regression model for classification where $\mathcal{Y} = \{0,1,...,K-1\}$, weights $\theta\in\mathbb{R}^{K\times d}$ and biases $b\in \mathbb{R}^K$. Given an input $x \in \mathbb{R}^d$, the model $f(x;\theta)$ produces a probability of each possible outcome in $\mathcal{Y} $ :
$$
f(x;\theta,b)=F_{\text{softmax}}(\theta x+b),
\\F_{\text{softmax}}(z)=\frac{1}{\sum_{k=0}^{K-1}e^{z_k}}(e^{z_0},e^{z_1},...e^{z_{K-1}})
$$
where $z_k$ is the $k$-th element of the vector $z$. The set of probabilities on $\mathcal{Y}$ is $P(\mathcal{Y}) := \{p ∈ \mathbb{R}^K :􏰋\sum^{K−1}_{k=0} p_k =1,p_k \ge 0\ \forall\ k=0,…,K-1\}$ The function $F_{\text{softmax}}(z):\mathbb{R}^K\rightarrow \mathcal{P(Y)}$ is called the “softmax function” and is frequently used in deep learning. $F_{\text{softmax}}(z)$ takes a K-dimensional input and produces a probability distribution on $\mathcal{Y}$. That is, the output of $F_{\text{softmax}}(z)$ is a vector of probabilities for the events $0, 1, . . . , K-1$. The softmax function can be thought of as a smooth approximation to the argmax function since it pushes its smallest inputs towards 0 and its largest input towards 1. 

### Objective Function

The objective function is the negative log-likelihood (commonly referred to in machine learning as the “cross-entropy error”): 
$$
\mathcal{L}(\theta,b)=\mathbb{E}_{(X,Y)}[\rho(f(X;\theta,b),Y)],
\\\rho(z,y)=-\sum_{k=0}^{K-1}\mathbf{1}_{y=k}\ \text{log}z_k,
\\\text{where }z_k \text{ is the k-th element of the vector }z\text{ and }\mathbf{1}_{y=k}\text{ is the indicator function}
\\

\mathbf{1}_{y=k}=\left\{ \begin{array}{ll}
1 & y=k\\
0 & y\ne k
\end{array} \right.
$$

### Mini-batch Gradient descent

$$
\theta^{(l+1)} = \theta^{(l)} -\alpha^{(l)}\frac{1}{M}\sum_{m=1}^M \nabla_\theta\rho(f(x^{(l,m)};\theta^{(l)},b^{(l)}),y^{l,m}),
\\b^{(l+1)} = b^{(l)} -\alpha^{(l)}\frac{1}{M}\sum_{m=1}^M \nabla_b\rho(f(x^{(l,m)};\theta^{(l)},b^{(l)}),y^{l,m})
$$

### Derivation of SGD

We will derive the stochastic gradient descent algorithm for the logistic regression model. The logistic regression model $f(x;\theta )$ is estimated from the dataset $(x_n, y_n)^N_{n=1}$ where $(x_n, y_n) ∼ \mathbb{P}_{X,Y}$ .

The gradient of the loss function for a generic data sample $(x, y)$ is
$$
\nabla_\theta \rho(f (x;\theta,b), y) = -\nabla_\theta \text{log}F_{\text{softmax},y} (\theta x+b)
$$
where $F_{softmax,k}(z)$ is the k-th element of the vector output of the function $F_{softmax,k}(z)$. Let $\theta_{k,:}$ be the k-th row of the matrix $θ$. 

If $k\ne y$, 
$$
\begin{eqnarray}
\nabla_{\theta_{k,:}}\rho(f (x;\theta ,b), y) & = & -\nabla_{\theta_{k,:}}\text{log}F_{\text{softmax},y} (\theta x+b)

\\ & = & -\frac{\nabla_{\theta_{k,:}}F_{\text{softmax},y}(\theta x+b)}{F_{\text{softmax},y}(\theta x+b)}

\\ & = & -\frac{1}{F_{\text{softmax},y}(\theta x+b)}\frac{0-e^{\theta_{y,:}x+b}\ e^{\theta_{k,:}x+b}}{(\sum_{m=0}^{K-1}e^{\theta_{m,:}x+b})^2}x

\\ & = & \frac{1}{F_{\text{softmax},y}(\theta x+b)}F_{\text{softmax},y}(\theta x+b)F_{\text{softmax},k}(\theta x+b)x\\ & = & F_{\text{softmax},k}(\theta x+b)x
\end{eqnarray}
$$
If $k=y$,
$$
\begin{eqnarray}
\nabla_{\theta_{k,:}}\rho(f (x;\theta,b), y) & = & -\nabla_{\theta_{k,:}}\text{log}F_{\text{softmax},y} (\theta x+b)
\\ & = & -\frac{\nabla_{\theta_{k,:}}F_{\text{softmax},y}(\theta x+b)}{F_{\text{softmax},y}(\theta x+b)}
\\ & = & -\frac{1}{F_{\text{softmax},y}(\theta x+b)}\frac{(\sum_{m=0}^{K-1}e^{\theta_{m,:}x+b})e^{\theta_{k,:}x+b}-e^{\theta_{y,:}x+b}\ e^{\theta_{k,:}x+b}}{(\sum_{m=0}^{K-1}e^{\theta_{m,:}x+b})^2}x
\\ & = & \frac{1}{F_{\text{softmax},y}(\theta x+b)}(1-F_{\text{softmax},k}(\theta x+b))F_{\text{softmax},y}(\theta x+b)x
\\ & = & F_{\text{softmax},k}(\theta x+b)x-x
\end{eqnarray}
$$


Hence, for any k,		
$$
\nabla_{\theta_{k,:}}\rho(f (x;\theta,b), y) = -(\mathbf{1}_{y=k}-F_{\text{softmax},y}(\theta x+b))x
$$
Therefore,
$$
\nabla_{\theta}\rho(f (x;\theta,b), y) = -(e(y)-F_{\text{softmax}}(\theta x+b))x^\top
$$
where,
$$
e(y)=(\mathbf{1}_{y=0},...,\mathbf{1}_{y=K-1})^\top.
$$
Similarly, we could get the gradient of the loss function with respect to $b$:
$$
\nabla_{b}\rho(f(x;\theta,b), y) = -(e(y)-F_{\text{softmax}}(\theta x+b))
$$
The stochastic gradient descent algorithm is:

- Randomly initialize the parameter $\theta^{(0)}, b^{(0)}$.

- For $\mathcal{l}=0,1,...,L$ :

  - Select $M$ data samples $(x^{(l,m)}, y^{(l,m)})^M_{m=1}$ at random from the dataset $(x_n, y_n)^N_{n=1}$, where $M \ll N$.
  - Calculate the gradient for the loss from the data sample:

  $$
  G_\theta^{(l)} = -\frac{1}{􏰌M}\sum_{m=1}^M (e(y^{(l,m)})-F_{\text{softmax}}(\theta^{(l)} x^{(l,m)}+b^{(l)}))(x^{(l,m)})^\top\\G_b^{(l)} = -\frac{1}{􏰌M}\sum_{m=1}^M (e(y^{(l,m)})-F_{\text{softmax}}(\theta^{(l)} x^{(l,m)}+b^{(l)}))
  $$

  - Update the parameters:

  $$
  \theta^{(l+1)}=\theta^{(l)}-\alpha^{(l)}G_\theta^{(l)},\\
  b^{(l+1)}=b^{(l)}-\alpha^{(l)}G_b^{(l)}
  $$

  ​	where $\alpha^{(l)}$ is the learning rate.
