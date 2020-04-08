import distribution_module as dist
import decision_boundary as disc
import matplotlib.pyplot as plt
import neural_network_module as nn
import numpy as np

epsilon = 0.01
x_min = -3.0
x_max = 3.0
y_min = -3.0
y_max = 3.0

syn_data = dist.SyntheticData()
syn_data.set_samples(200)
DBoundary = disc.D2_boundary(syn_data.f, epsilon, x_min, x_max, y_min, y_max)
X_train, target_train, color_dummy, n_samples = syn_data.gen_data()
X_val, target_val, color, n_samples = syn_data.gen_data()
tau = 150
network = nn.SimpleNetwork()
network.set_h_unit(6)
network.learn(X_train, target_train, n_samples, tau)
NNBoundary = disc.D2_boundary(network.f, epsilon, x_min, x_max, y_min, y_max)

print(network.error_rate(X_val, target_val, n_samples), '%')
plt.scatter(X_val[:,0], X_val[:,1], c = color)
plt.scatter(DBoundary[:,0], DBoundary[:,1], c='green', s=0.5)
plt.scatter(NNBoundary[:,0], NNBoundary[:,1], c='black', s=0.5)
plt.show()
