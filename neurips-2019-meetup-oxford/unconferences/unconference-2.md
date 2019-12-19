---
description: >-
  Some notes on the second unconference, which consists of meta learning and
  first/second order neural network optimizers
---

# Unconference 2

## Part I: What is Meta Learning?

### Goal of meta-learning:

* To train a meta-learner \(often a neural network\) in order to enable fast learning of a new task given few examples using a new neural network
* This is useful in cases where training examples are scarce

### What does a meta-learning dataset look like?

![Visualizing the dataset used in Meta Learning](../../.gitbook/assets/img_6182.JPG)

* train on similar tasks $$T_1, ...,T_n$$, where each task consists of a:
  1. context dataset, namely task data for training the neural network
  2. target dataset, namely test data for training the neural network
     * this dataset is unlabelled when supplied to the meta-learner

  * note that the size of the context dataset is very small
* an example in the context of the MNIST handwritten digit dataset
  * create 100 tasks, where each task consists of 5 randomly sampled labelled examples and 5 randomly sampled unlabelled examples, which form the context and target dataset respectively
    * where each example comes from a different class
  * goal is to use meta-learner to achieve few-shot learning for a new task $$T_{101}$$ 
* * note that there are multiple ways of achieving this
  * for example, one can learn the best learning rate for a group of similar tasks
  * alternatively, one can learn the best weight initialization scheme instead

### A simple example

* a simple/basic example of meta-learning
  * e.g. find best learning rate $$\alpha$$ for a particular task $$T_i$$ 
  * i.e. train a neural network $$f_\phi : T_i \rightarrow \alpha$$ where $$T_i$$'s context and target dataset is the neural network's input
* meta learning often involves a for loop inside a for loop
  * outer loop for training $$f_\phi$$ , inner loop for training the neural network for a specific task
* main examples of meta learning:
  * [Learning to learn by gradient descent by gradient descent](https://papers.nips.cc/paper/6461-learning-to-learn-by-gradient-descent-by-gradient-descent.pdf)
  * Model-agnostic model learning \(MAML\)

## Optimizers 

### Gradient Descent is a first order optimizer

* weight update equation: $$\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$$ 
  * for weights $$\theta_{t}, \theta_{t+1}$$ at time steps $$t$$ and $$t+1$$ respectively
  * and loss function $$L(\theta)$$ 
  * where $$\alpha$$ and $$\nabla L(\theta_t)$$ are the learning rate and gradient respectively
* using a first order approximation, this update equation can be equivalently written as

  $$\theta_{t+1} - \theta_t =  \text{argmin}_{\delta} \{L(\theta_t) + \nabla L(\theta_t)^\intercal\delta + \frac{1}{2} \delta^2\}$$ 

  * note: solving for $$\alpha$$ in terms of $$\delta$$ yields $$\alpha = - \delta$$ 

* breaking down the terms inside the argmin:
  * gradient descent is a first order optimizer because it uses a first order Taylor approximation of the loss function, namely $$L(\theta_t+\delta) \approx L(\theta_t) + \nabla L(\theta_t)^\intercal\delta$$ 
  * the $$\frac{1}{2} \delta^2$$ term ensures "localness" of minimization, i.e. prevents $$\theta_{t+1}$$ from moving too far away from $$\theta_t$$ 

### Improving gradient descent slightly

* $$L(\theta)$$ is often nonnegative
  * e.g. for mean squared error or negative log likelihood loss functions
* a problem arises when a large$$\alpha$$causes the term $$L(\theta_t) + \nabla L(\theta_t)^\intercal\delta$$ to become negative, which is an inaccurate estimate of $$L(\theta_t+\delta)$$ !
* to solve this, we use a max function to bound the first-order approximation, which results in the following update equation:

  $$\theta_{t+1} - \theta_t =  \text{argmin}_{\delta} \{\text{max}(L(\theta_t) + \nabla L(\theta_t)^\intercal\delta, 0) + \frac{1}{2} \delta^2\}$$ 

* this modified version of gradient descent gives the same or slightly better results than plain gradient descent

### Newton's Method

* uses a second order Taylor expansion instead of a first order one, which gives the update equation

  $$\theta_{t+1} - \theta_t =  \text{argmin}_{\delta} \{L(\theta_t) + \nabla L(\theta_t)^\intercal\delta + \frac{1}{2} \delta^\intercal H(\theta_t)\delta + \frac{1}{2} \delta^2\}$$ 

  where $$H(\theta)$$ is the Hessian, i.e. $$H_{ij} = \frac{\partial^2 L}{\partial \theta^i \theta^j}$$ for weights $$\theta^i, \theta^j$$ 

* to reach the local minimum, we need $$0 = \nabla L(\theta) + H(\theta)\delta$$ 
  * i.e. $$ \delta = -[H(\theta)]^{-1} \nabla L(\theta)$$ 
* A big limitation of this method is that the inversion of the Hessian is expensive to compute \(it takes cubic time\)

### Natural Gradient Descent

* use Fisher matrix $$F$$ to replace Hessian, which gives the update equation  $$\theta_{t+1} - \theta_t =  \text{argmin}_{\delta} \{L(\theta_t) + \nabla L(\theta_t)^\intercal\delta + \frac{1}{2} \delta^\intercal F(\theta_t)\delta + \frac{1}{2} \delta^2\}$$ 
* Since $$F = -\mathbb{E}_{p(x|\theta)}[H_{\log p(x|\theta)}] = H_{\text{KL}[p(x|\theta) \parallel p(x|\theta') ]} |_{\theta' = \theta}$$ , we can do the following:
  * By using $$F$$ , we are taking a second-order derivative of the KL-divergence $$\text{KL}[p(x|\theta) \parallel p(x|\theta')]$$ at $$\theta = \theta'$$ instead of the loss function $$L(\theta)$$ 

    =&gt; minimizing in $$p(x|\theta)$$ space instead of parameter space 

  * we can perform a second order Taylor series expansion of $$\text{KL}[p(x|\theta) \parallel p(x|\theta')]$$ at $$\theta = \theta'$$**:**

    \*\*\*\*$$\text{KL}[p(x|\theta) \parallel p(x|\theta')] |_{\theta' = \theta} \\ = \text{KL}[p(x|\theta) \parallel p(x|\theta)] + (\nabla_{\theta '} \text{KL}[p(x|\theta)\parallel p(x|\theta')] |_{\theta'=\theta})^\intercal \delta + \frac{1}{2}\delta^\intercal F \delta + O(\delta^3) \\ = \frac{1}{2}\delta^\intercal F \delta + O(\delta^3) \\ \approx \frac{1}{2}\delta^\intercal F \delta$$ ****

    * the second to last step comes from the fact that:
      1. $$\text{KL}[p(x|\theta) \parallel p(x|\theta)] = 0 $$ since $$\text{KL}[p \parallel q] = 0 $$ for $$p = q$$ 
      2. $$\nabla_{\theta'} \text{KL}[p(x|\theta)\parallel p(x|\theta')] |_{\theta = \theta'}$$   
         $$= \nabla_{\theta'} \mathbb{E}_{p(x|\theta)}[\log p(x|\theta')] |_{\theta = \theta'}$$   
         $$= \mathbb{E}_{p(x|\theta)}[\nabla_{\theta'}\log p(x|\theta')|_{\theta = \theta'}]$$ 

         $$= \mathbb{E}_{p(x|\theta)}[\nabla_{\theta}\log p(x|\theta)]$$   
         $$= $$ $$\int p(x|\theta) \nabla_{\theta} \log p(x|\theta) dx$$ 

         $$= \int p(x|\theta) \frac{\nabla_{\theta} p(x|\theta)}{p(x|\theta)} dx$$ 

         $$= \int \nabla_{\theta} p(x|\theta) dx$$ 

         $$= \nabla_{\theta} \int  p(x|\theta) dx$$ 

         $$= \nabla_{\theta} 1$$ 

         $$= 0$$ 

  * thus, the argmin term in the weight equation for natural gradient descent can be seen as the summation of:
    1. a first order approximation of $$L(\theta_t + \delta)$$ 
    2. some KL regularization to keep the new distribution $$p(x|\theta')$$ close to the original distribution $$p(x|\theta)$$ 
    3. $$\frac{1}{2}\delta^2$$ term to keep $$\theta_{t+1}$$ close to $$\theta_t$$ \(i.e. stay in local region\)
* similar to Newton's method, we need $$ \delta = -[F(\theta)]^{-1} \nabla L(\theta)$$ for the local minimum
* note: computing the actual Fisher matrix via MC sampling from the distribution is better than using empirical Fisher matrix
* However inverting the Fisher matrix also takes cubic time

  * moreover the Hessian need not be +ve definite in a neural network

  =&gt; we want to estimate the Hessian or Fisher matrix

### K-FAC

* stands for Kronecker Factorization
* uses Kronecker products to estimate the hessian for each neural network layer
  * these products have some nice properties :\)
* bad news: method is layer-type dependent
  * weaker guarantees for convolution or recurrent layers compared to fully connected layers
* works well in practice for specific architectures \(details in paper\)
* Question: is there a spectrum between K-FAC and the use of the Fisher matrix?



