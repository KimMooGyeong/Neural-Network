# Neural-Network-NN_Classification


This program is an implementation of neural network to classify a synthetic data set created by mixture of Gaussian.

Explanation of files:
  - distribution_module.py contains class 'DataModule' to generate two dimensional synthetic data set whose target data is 1 or 0.
  - decision_boundary.py contains functions to plot f(x, y) = 0 in two dimensional space, where function f is arbitrary.
  - neural_network_module.py has class 'SimpleNetwork' which has one hidden layer. Number of unit in hidden layer can be modified
    by simply changing the member of class self._M, or using method self.set_h_unit(m). Number of input and output unit is fixed.
  - main.py import above three files and plot the decision boundary of neural network and optimal boundary. It also print the
    error rate evaluated by (Number of data points that is not classified correctly)/(Total number of data points)
  - error_plot.py has several plotting mode. You can modify the global variable plot_num to plot different graph.
