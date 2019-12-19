---
description: Establishes various frameworks to allow machine learning to advance science
---

# Keynote: Veridical Data Science

## Links

Presentation: [https://slideslive.com/38921720/veridical-data-science](https://slideslive.com/38921720/veridical-data-science)

## PCS Framework

### What is this?

* a framework for predictability, computability and stability
  * predictability and computability comes from machine learning\
    * computability often involves simulations
  * stability comes from statistics
    * often includes replicability
* Stability is mostly on perturbation analysis
* data cleaning processes is under-reported, e.g. for RNA-seq data
* Often times in science we make decisions when performing experiments
  * we need to make each judgement call clear so we can easily reproduce things
* P is for the future, not just for one day
* S is also for modelling

### Perturbations

What type of decisions might we make when trying to do science in machine learning?

#### Data perturbations

* data modalilty choices
* synthetic data \(e.g. from PDE models\)
* from invariance
* differential privacy
* adversarial attacks to deep learning algorithms \(e.g. to medical images\)
* DATA CLEANING!

#### Existing model or algorithm perturbations

* the use of lasso vs ridge models
* modes of a non-convex empirical minimization
* robust statistics
* semi-parametrics
* kernels
* sensitivity analysis
* RESEARCHER perturbation

### Documentation is important

* detailed documentation on every judgement calls
* this is especially important on Github!
  * currently it is not uncommon to find published papers with code that don't quite run on Github
* this is how we connect the bridge between reality and models

#### How to choose perturbations in PCS?

* we need legitimate perturbations
* therefore, document appropriateness of all perturbations made
  * any appropriations will do, they need not be well justified scientifically
  *  \(e.g. "I only know this method" is okay!\)

### Statistical inference under PCS

* using scientific evidence for making decisions
* p-value is problematic only when using the wrong model
  * using a wrong model and getting negative results does not imply that the hypothesis is wrong!
* data is the realization of a process
* it is an assumption unless it is explicit randomization
* why think about the future?
* p-values measure model bias
  * there is no "true model"

### Inference beyond probabilistic models \(e.g. via PDE\)

1. Problem formulation
   * this is independent of the algorithm used
2. Prediction screening for reality check
   * e.g. cross validation \(as simple as it is, it's better than nothing\)
3. Target value perturbation distribution
   * involves data cleaning and model perturbation
4. Summarize

Post-selection isn't always needed

## Making Random Forest more stable

* want to do pattern discovery

### iterative Random Forest

* the use of the french flag model results in the threshold behaviour captured by decision trees
* involves iteratively reweighing random forests

### Random Intersection Trees

* like market basket problem
* e.g. use in enhancer activity in Drosophila
* a stable intersection means new recommendation for new experiments

Side note: there's a really nice book called "Interpretable Machine Learning" by Christoph Molnar

## Definitions, methods, and applications in interpretable machine learning

### PDR framework 

* for interpretable machine learning \(iML\)
* P = prediction, D = descriptive, R = relevancy

### Agglomerative contextual decomposition for interpreting neural networks

* before: gradient-based and contextual-based methods
* contextual decomposition applied to cosmology

## Conclusion

* currently there's no certification or labelling of ML algorithms
  * hard to tell which one is better than the other
* therefore we need **trustworthy publications**
  * quality over quantity
  * this new practice needs to start from the community

