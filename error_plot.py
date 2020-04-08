import distribution_module as dist
import decision_boundary as disc
import neural_network_module as nn

import numpy as np
import matplotlib.pyplot as plt

syn_dist = dist.SyntheticData()
network = nn.SimpleNetwork()
M, N, tau = 10, 1000, 500
# number of error sampling for each case
samp = 50
"""plot_num determine the x-axis"""
plot_num = 2
# plot_num is 1 for plot of Error versus number of samples
if plot_num == 1:
    error_data = []
    for m in range(1, M):
        errors = []
        for n in range(10, N, 25):
            syn_dist.set_samples(n)
            network.set_h_unit(m)
            err = 0
            for i in range(samp):
                X_train, T_train, color, n_samples = syn_dist.gen_data()
                X_val, T_val, color, n_samples = syn_dist.gen_data()
                network.learn(X_train, T_train, n_samples, tau)
                err += network.error_rate(X_val, T_val, n_samples)
                print(err)
            errors.append(err/samp)
        error_data.append(errors)
    sample_num = list(range(10, N, 25))

    plt.title('Error vs. num_sample')
    plt.xlabel('num_sample')
    plt.ylabel('Error rate')

    plt.plot(sample_num, error_data[0], marker='s', c='RED')
    plt.plot(sample_num, error_data[1], marker='s', c='ORANGE')
    plt.plot(sample_num, error_data[2], marker='s', c='GREEN')
    plt.plot(sample_num, error_data[3], marker='s', c='BLUE')
    #plt.plot(sample_num, error_data[4], marker='s',c='BLACK')

    plt.show()

# plot_num is 2 for plot of error of test and training set
if plot_num == 2:
    error_train = []
    error_test = []
    model_complexity = []
    syn_dist.set_samples(int(N))
    X_train, T_train, color, n_samples_train = syn_dist.gen_data()
    syn_dist.set_samples(int(N*3/7))
    X_test, T_test, color, n_samples_test = syn_dist.gen_data()
    for m in range(1, M):
        network.set_h_unit(m)
        err_train = 0
        err_test = 0
        for i in range(samp):
            network.learn(X_train, T_train, n_samples_train, tau)
            temp_train = network.error_rate(X_train, T_train, n_samples_train)
            temp_test = network.error_rate(X_test, T_test, n_samples_test)
            print(temp_train, temp_test)
            err_train += temp_train
            err_test += temp_test
        model_complexity.append(m)
        error_train.append(err_train/samp)
        error_test.append(err_test/samp)

    plt.title('Error vs. complexity')
    plt.xlabel('model complexity')
    plt.ylabel('Error rate')

    plt.plot(model_complexity, error_train, marker='s', c='RED')
    plt.plot(model_complexity, error_test, marker='s', c='BLUE')
    plt.show()

# plot_num is 3 for scattering error versus model complexity
if plot_num == 3:
    print(plot_num)
    syn_dist.set_samples(int(N * 0.7))
    X_train, T_train, color, n_samples_train = syn_dist.gen_data()
    syn_dist.set_samples(int(N * 0.3))
    X_test, T_test, color, n_samples_test = syn_dist.gen_data()
    for m in range(1, M):
        network.set_h_unit(m)
        error = []
        model_complexity = [m] * samp
        for i in range(samp):
            network.learn(X_train, T_train, n_samples_train, tau)
            temp = network.error_rate(X_test, T_test, n_samples_test)
            error.append(temp)
            print(temp)
        plt.scatter(model_complexity, error, marker='x', c='BLUE')

    plt.title('Error vs. complexity')
    plt.xlabel('model complexity')
    plt.ylabel('Error rate')

    plt.show()

# plot4 is to see the effect of early stopping
if plot_num == 4:
    syn_dist.set_samples(int(N * 0.4))
    X_train, T_train, color, n_samples_train = syn_dist.gen_data()
    syn_dist.set_samples(int(N * 0.6))
    X_test, T_test, color, n_samples_test = syn_dist.gen_data()
    network.set_h_unit(M)
    error = []
    for t in range(10, 1000, 30):
        err = 0
        for i in range(samp):
            network.learn(X_train, T_train, n_samples_train, t)
            error += network.error_rate(X_test, T_test, n_samples_test)
        error.append(err)

    plt.title('Error vs. iteration num')
    plt.xlabel('number of iteration')
    plt.ylabel('Error rate')
    plt.plot(list(range(10, 1000, 30)), error, marker='r', c='BLUE')
    plt.show()

# plot5 is to find the proper learning rate
if plot_num == 5:
    syn_dist.set_samples(int(N*0.7))
    X_train, T_train, color, n_samples_train = syn_dist.gen_data()
    syn_dist.set_samples(int(N*0.3))
    X_test, T_test, color, n_samples_test = syn_dist.gen_data()

    etha0, etha_step = 0.001, 0.0005
    num_step = 20
    ethas = [etha0 + i * etha_step for i in range(num_step)]
    err = []

    network.set_h_unit(4)

    for etha_current in ethas:
        if network.is_set():
            network.learn(X_train, T_train, n_samples_train, tau)
        else:
            W10, W20 = network.learn(X_train, T_train, n_samples_train, tau)
            network.set_weight(W10, W20)
        temp = network.error_rate(X_test, T_test, n_samples_test)
        print('learning rate: ', etha_current, 'error: ', temp)
        err.append(temp)

    plt.plot(ethas, err, c="BLUE")
    plt.show()
