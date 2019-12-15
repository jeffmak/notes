---
description: >-
  Explores a framework, which involves which ensures fairness is created from
  representation learning
---

# Tutorial: Fairness and Representation Learning

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

is a good metric?





