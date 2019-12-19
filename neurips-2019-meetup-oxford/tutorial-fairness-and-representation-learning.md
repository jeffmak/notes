---
description: >-
  Explores a framework, which involves the data regulator, data producer and
  data user. Such a framework ensures that fairness is created from
  representation learning
---

# Tutorial: Fairness and Representation Learning

## Links

Recording: [https://slideslive.com/38921491/representation-learning-and-fairness](https://slideslive.com/38921491/representation-learning-and-fairness) 

Slides: [http://sanmi.cs.illinois.edu/documents/Representation\_Learning\_Fairness\_NeurIPS19\_Tutorial.pdf](http://sanmi.cs.illinois.edu/documents/Representation_Learning_Fairness_NeurIPS19_Tutorial.pdf)

## Introduction

### Problem and Goal

Currently there is no "one-definition" for fairness in machine learning

### Framework

We introduce a framework where there are three entities:

1. Data regulator:
   * works at the regulatory/high level
   * produces fairness criteria
   * performs auditing:
     * checks that fairness constraints are met by the representations used by the data user
   * uses fairness criteria to define a technical description on what qualifies as fair
     * includes score functions, fairness metrics and sensitive attributes
2. Data producer
   * takes the fairness criteria and creates the fair representation
3. Data user
   * computes the machine learning model and passes the model back to the data regulator for auditing

Note that the three entities form a closed loop, i.e., data regulator -&gt; data producer -&gt; data user -&gt; data regulator -&gt; ...

## Data Regulator

### How to define the fairness criteria?

#### 1. Individual Fairness

This is where similar individuals are treated similarly. For example, similar images should yield similar classifications.

This leads to the following question: which individual are similar or should be treated similarly? One solution to this is to partition the embedded space such that the similar individuals belong in the same cell or region.

Note that:

1. Individual fairness implies algorithmic robustness
2. Algorithmic robustness implies generalization, i.e., the difference between the training and test loss is bounded

Currently, it is challenging to create individually fair models with low training error and generalization.

Since Lipschitz continuity implies individual fairness, fairness can be achieved by Lipschitz regularization. However, most high-dimensional data, e.g. images, are non-euclidian. Then the question becomes the following:

Can we learn a representation of data such that

$$
\rho = || \cdot ||_2
$$

is a good metric for Lipschitz continuity?

#### 2. Group Fairness

* which statistic v\(f\|Y, s\) should be **equalized** across the groups
  * eg. can be measured by statistical parity \(i.e. computing the true positive rate and false positive rate\) using a confusion matrix
  * other measures include equal of opportunity and equalized odds
* confusion matrix only has two degrees of freedom
  * degree of freedom grows quadratically:
    * degree of freedom = \(no. of classes\)\*\*2 - no. of classes
  * =&gt; may not be easy to incorporate in training
* policy makers are more comfortable with group fairness since the concept of population is involved
* However, group fairness can lead to more violated individual fairness
  * e.g. intersectionality can lead to fairness gerrymandering

### **How to measure fairness then? Which metric to use?**

* there is no one absolute way to measure fairness
  * since a fairness measure ceases to be one when someone explicitly knows and exploits the measure
  * since all metrics have failure cases, we need to think about tradeoffs

### **Metric Elicitation** \(for group fariness\)

* need to pick good metric
* solution: query experts until a good metric is selected

### **Audit representation**

* need to audit the representation produced by the data producer
* sometimes also audit the representation used by the data user
  * especially if there is an adversarial user

Once a fairness criteria is created by the data regulator, the criteria gets passed to the data producer

## Data producer

* takes the fairness criteria and data and creates a representation
  * the representation created can be label-dependent or label-independent

### Representation learning

* transform data from high dimension to low dimension
* common representation learning algorithms include PCA and nonlinear autoencoders
  * we'll base on these and build up from these algorithms later

### For individual fairness:

* surprising there isn't much research work left to do here
* take sets of examples which should be treated similarly from data regulator, and learn a distance metric which satisfies the individual fairness properties
  * i.e. similar examples should be closer in distance than those with different examples
* Question: what notion of sets make sense?

### For group fairness

There are three things to be aware of, namely representation, prediction and fairness. The following shows representation learning algorithms which use group fairness:

#### Semi-supervised variational autoencoder and MMD fairness

* measures the distance between two distributions from each group
* fairness may improve performance \(see results in paper\)

#### Learning Controllable Fair Representations

* uses a variety of fairness measures
* uses information-theoretic approximation to fairness metrics
  * with performance, this gives nice properties in terms of representation learning

#### An adversarial approach for learning fair representation

* for disentangling
* uses a variety of fairness measures in specialized adversary loss function

#### **Tradeoff between performance and fairness**

* high accuracy may cause weird things... \(more on this later\)

#### General Adversarial Representations

* uses optimal adversary

#### **Fair Representations using Disentanglement**

* statistical parity
* penalizing mutual information fo sensitive and non-sensitive features for disentanglement
* can look at combination of features at test time =&gt; can pick and choose features

#### On the fairness of disentangled representations

* uses statistical parity, though it isn't mentioned in the paper
* no sensitive measure used, i.e. no regularization in measure

#### Conjecture: Disentangled representation results in fairness?

## Data user

* the data user constructs a machine learning model given a sanitized representation
* the data should not add new features to the machine learning model
* the ML model created should be audited by the data regulator

### Representation learning when considering the data user

Advantages:

* the representation created from representation learning can be reused over multiple tasks
* representation learning is good if you don't trust the data user
* representation learning is good for weak privacy constraints
  * i.e. if you don't want the data user to touch the input \(can get around this easily tho\)

Disadvantages:

* However, the use of representation learning results in less precise control of the fairness-performance tradeoff
  * may lead to fairness overconfidence
* startup costs can be high

## Conclusion

### A processing perspective

Another way of looking at this entire process is the following:

1. Pre-processing
   * involves representation learning
2. In-processing
   * involves joint learning and fairness regulation
3. Post-processing
   * involves adjustment of potentially unfair model \(e.g. threshold adjustment\)

Pre-processing and in-processing form a good basis for generalization

### But we've only scratched the surface

There are still many other problems to solve, such as:

* how to collect data in a thoughtful way
* how to avoid data leakage
* how to ensure algorithmic fairness for continuous variables
* how to create fairness for specific contexts

## Misc from QA session

* adversarial approaches often serve as a hammer for getting good fairness properties





