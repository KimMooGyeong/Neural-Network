import numpy as np
from numpy.random import randn
import numpy.linalg as lin
import distribution_module as dist

class SimpleNetwork:
    def __init__(self, M = 2, etha = 0.025):
        self._M = M
        self._etha = etha
        self._W1 = None
        self._W2 = None
        self._a1 = None
        self._z1 = None
        self._a2 = None
        self._y = None
        self._gradE1 = np.zeros((M,3))
        self._gradE2 = np.zeros(M).reshape((1,M))
        self._is_set = False

    """Initialize weights"""
    def _init_weight(self):
        #self._W1 = (30.0 * np.random.rand(self._M * 3) - 15.0 * np.ones(self._M * 3)).reshape((self._M, 3))
        #self._W2 = (30.0 * np.random.rand(self._M) - 15.0 * np.ones(self._M)).reshape((1,self._M))
        self._W1 = (20.0 * randn(self._M, 3)).reshape((self._M, 3))
        self._W2 = (20.0 * randn(self._M)).reshape((1, self._M))
        return

    """Init gradient E"""
    def _init_grad(self):
        self._gradE1 = np.zeros((self._M,3))
        self._gradE2 = np.zeros(self._M).reshape((1,self._M))

    """Function definitions"""
    def _sig(self, x):
        return np.divide(1.0 , (1 + np.exp(-x)))

    def _tanh(self, x):
        return 2.0 * self._sig(2.0*x) - 1.0

    def _sigp(self, x):
        return self._sig(x) * (1.0 - self._sig(x))

    def _tanhp(self, x):
        return 1.0 - (self._tanh(x) * self._tanh(x))

    """Forward propagation methods"""
    def _eval_a1(self, x):
        return self._W1 @ x.T

    def _eval_z1(self, a1):
        return self._tanh(a1)

    def _eval_a2(self, z1):
        return (self._W2 @ z1)

    def _eval_output(self, a2):
        return self._sig(a2)

    def output(self):
        return self._y

    def _forward(self, x):
        self._a1 = self._eval_a1(x)
        self._z1 = self._eval_z1(self._a1)
        self._a2 = self._eval_a2(self._z1)
        self._y = self._eval_output(self._a2)
        return

    """Back Propagation methods"""
    def _error1(self, x, t):
        delta = self._y - t
        vec1 = x      # 1 by 3 vector
        vec2 = self._tanhp(self._a1) * self._W2.T    # M by 1 vector
        cons = delta * self._sigp(self._a2)
        return cons * (vec2 @ vec1) # return M by 3 matrix

    def _error2(self, x, t):
        delta = self._y - t
        return delta * (self._z1.T)

    """Batch gradient descent"""
    def _batchE(self, x, t):
        # forward propagation
        self._forward(x)
        # back propagation
        error1 = self._error1(x, t)
        error2 = self._error2(x, t)
        return error1, error2

    def _backprop(self, data_set, target, n):
        i = 0
        one = np.array([1])
        self._init_grad()
        while i < n:
            x = np.concatenate((one, data_set[i]), axis=0).reshape((1,3))
            t = target[i]
            error1, error2 = self._batchE(x, t)
            self._gradE1 += error1
            self._gradE2 += error2
            i += 1
        self._W1 -= (self._etha * self._gradE1)
        self._W2 -= (self._etha * self._gradE2)
        return

    """Public methods"""
    # modify the number of hidden units
    def set_h_unit(self, n):
        print('evaluate M=', n)
        self._M = n
        return

    def get_h_num(self):
        m = self._M
        return m

    # modify learning rate
    def set_learning_rate(self, eta):
        print('evaluate eta = ', eta)
        self._etha = eta
        return

    # set initial weight
    def set_weight(self, W10, W20):
        self._is_set = True
        self._W10, self._W20 = W10, W20
        return

    # query initial weight is set
    def is_set(self):
        return self._is_set

    # turn off _is_set
    def set_turn_off(self):
        self._is_set = False
        return

    # method for learning
    def learn(self, data_set, target, n, tau):
        i = 0
        if self._is_set:
            pass
        else:
            self._init_weight()
        W10, W20 = self.get_weight()
        for i in range(tau):
            self._backprop(data_set, target, n)
        return W10, W20

    # method to obtain weight
    def get_weight(self):
        return self._W1, self._W2

    # method to obtain error rate
    def error_rate(self, data_set, target, n):
        result = 0.0
        one = np.array([1])
        i = 0
        while i < n:
            x = np.concatenate((one, data_set[i]), axis=0).reshape((1,3))
            # foward propagation
            self._forward(x)
            if self.output() > 0.5:
                y = 1.0
            else:
                y = 0.0
            result += abs(y - target[i])
            i += 1
        return result * 100.0 / n

    # function to plot decision boundary
    def f(self, x1, x2):
        result = 0.0
        for j in range(self._M):
            aj = self._W1[j][0] + self._W1[j][1] * x1 + self._W1[j][2] * x2
            w2 = self._W2[0][j]
            result += w2 * self._tanh(aj)
        return result

"""Test"""
if __name__ == '__main__':
    #syn_data = dist.SyntheticData()
    X, T, color, n_samples = dist.gen_data()
    tau = 300
    network = SimpleNetwork()
    network.set_h_unit(3)
    network.learn(X, T, n_samples, tau)
    print(network.get_weight())
    print(network.error_rate(X, T, n_samples), '%')
