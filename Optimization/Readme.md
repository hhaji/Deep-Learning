# Optimization

## [Hyperparameter (machine learning)](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))   

A model parameter is a variable whose value is estimated from the dataset. Parameters are the values learned during training from the historical data sets like weights and biases. A hyperparameter is a configuration variable that is external to the model. It is defined manually before the training of the model with the historical dataset. Its value cannot be evaluated from the datasets. For example, the number of hidden nodes and layers,input features, learning rate, activation function etc in neural network.  

In machine learning, the value of a hyperparameter is used to control the learning process. By contrast, the values of parameters (typically edge weights) 
are derived via training. 

## [Optimizing Hyperparameters](https://deepai.org/machine-learning-glossary-and-terms/hyperparameter)  
Hyperparameters can have a direct impact on the training of machine learning algorithms. Thus, in order to achieve maximal performance, 
it is important to understand how to optimize them. Here are some common strategies for optimizing hyperparameters:

- Grid Search: Search a set of manually predefined hyperparameters for the best performing hyperparameter. Use that value. (This is the traditional method)   

- Random Search: Similar to grid search, but replaces the exhaustive search with random search. This can outperform grid search when only a 
small number of hyperparameters are needed to actually optimize the algorithm.   

- Bayesian Optimization: Builds a probabilistic model of the function mapping from hyperparameter values to the target evaluated on a validation set.   

- Gradient-Based Optimization: Compute gradient using hyperparameters and then optimize hyperparameters using gradient descent.   

- Evolutionary Optimization: Uses evolutionary algorithms (e.g. genetic functions) to search the space of possible hyperparameters.   

## [A Novice’s Guide to Hyperparameter Optimization at Scale](https://wood-b.github.io/post/a-novices-guide-to-hyperparameter-optimization-at-scale/)  
by Brandon M. Wood   

There are two main types of hyperparameters in machine learning and they dictate what HPO strategies are possible.

- Model (Structural) Hyperparameters: Establish model architecture    
**Structural hyperparameters** refer to the model selection task and they cannot be inferred while fitting the machine to the training 
set. For example: Number of convolutional layers, Number of fully connected layers, etc.

- Algorithm Hyperparameters: Are involved in the learning process   
**Algorithm hyperparameters**, that in principle have no influence on the performance of the model but affect the speed and quality of the learning process.  Examples of algorithm hyperparameters are learning rate and mini-batch size.  

**Not all Hyperparameters Can Be Treated the Same:** 
The important takeaway is that not all HPO strategies can handle both structural and algorithm hyperparameters. Population basd training (PBT) is a good example. PBT was designed to evolve and inherit hyperparameters from other high performing workers in the population; however, if workers have different network architectures it is unclear how exactly that would work. There might be a way to do this with PBT, but it is not standard and does not work out-of-the-box with Ray Tune.   
