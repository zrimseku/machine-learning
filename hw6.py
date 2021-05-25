import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def cross_entropy_cost(net, X, y):
    _, A = forward(net, X)
    Y = A[-1]
    ce = sum(np.log(Y[y, range(len(y))]))
    if net.act_last == 'sigmoid':
        for i in range(len(y)):
            Y = A[-1][:, i]
            t = np.zeros(Y.shape[0])
            t[y[i]] = 1
            ce += np.dot(1 - t, np.log(1 - Y))
    elif net.act_last != 'softmax':
        print('Use softmax, sigmoid or linear for last layer activation.')
    return -ce / len(y)


def mse_cost(net, X, y):
    _, A = forward(net, X)
    mse = sum((A[-1][0, :] - y) ** 2) / len(y) / 2
    return mse


def cost(net, X, y):
    if net.act_last == 'linear':
        f = mse_cost(net, X, y)
    else:
        f = cross_entropy_cost(net, X, y)
    return f


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(net, X):
    Z = [X]
    A = [X]
    for i in range(net.nr_layers - 2):
        z = net.weight[i] @ X + net.bias[i]
        if net.activation == 'sigmoid':
            X = sigmoid(z)
        elif net.activation == 'relu':
            X = np.maximum(z, 0)
        else:
            print('Unknown activation function. Use sigmoid or relu.')
            exit(-1)
        Z.append(z)
        A.append(X)

    # Output layer
    z = net.weight[-1] @ X + net.bias[-1]
    Z.append(z)
    if net.act_last == 'sigmoid':
        A.append(sigmoid(z))
    elif net.act_last == 'softmax':
        softmax = np.exp(z)
        softmax /= np.sum(softmax, axis=0)
        A.append(softmax)
    elif net.act_last == 'linear':
        A.append(z)
    else:
        print('Use softmax, sigmoid or linear for last layer activation.')
        exit(-1)

    return Z, A


def backpropagation(net, X, y):
    Z, A = forward(net, X)
    dW = [np.zeros(np.shape(w)) for w in net.weight]
    db = [np.zeros(np.shape(b)) for b in net.bias]

    dz = A[-1]
    if net.act_last == 'linear':
        dz -= y
    else:
        dz[y, range(len(y))] += -1
    db[-1] = np.reshape(np.sum(dz, axis=1), (net.layer_sizes[-1], 1)) / len(y)
    dW[-1] = dz @ A[-2].T / len(y)

    for i in range(2, net.nr_layers):
        if net.activation == 'sigmoid':
            act_derivative = A[-i] * (1 - A[-i])
        elif net.activation == 'relu':
            act_derivative = Z[-i] > 0
        else:
            print('Unknown activation function. Use sigmoid or relu.')
            exit(-1)
        dz = act_derivative * (net.weight[1 - i].T @ dz)
        dW[-i] = dz @ A[-i-1].T / len(y)
        db[-i] = np.reshape(np.sum(dz, axis=1), (dz.shape[0], 1)) / len(y)
    return dW, db


def update_network(net, weight_bias_vec):
    currently_on = 0
    for i in range(len(net.layer_sizes) - 1):
        l1, l2 = net.layer_sizes[i], net.layer_sizes[i + 1]
        net.weight[i] = weight_bias_vec[currently_on:currently_on + l1 * l2].reshape((l2, l1))
        currently_on += l1 * l2
        net.bias[i] = weight_bias_vec[currently_on:currently_on + l2].reshape((l2, 1))
        currently_on += l2
    return net.weight, net.bias


class ANNClassification:

    def __init__(self, units, n_input=None, n_classes=None, lambda_=0, activation_fn='sigmoid', act_last='sigmoid'):
        # number of input parameters and classes are set to 1, if we dont know the values at initialization
        self.nr_layers = len(units) + 2
        self.layer_sizes = [n_input, *units, n_classes]
        self.bias = None
        self.weight = None
        self.reg = lambda_
        self.activation = activation_fn
        self.act_last = act_last

    def cost_for_optimization(self, weight_bias_vec, X, y):
        # change weights and bias of current net to new ones
        self.weight, self.bias = update_network(self, weight_bias_vec)
        # return cost
        return cross_entropy_cost(self, X, y)

    def backprop_for_optimization(self, weight_bias_vec, X, y):
        # change weights and bias of current net to new ones
        self.weight, self.bias = update_network(self, weight_bias_vec)
        dW, db = backpropagation(self, X, y)
        return np.hstack([np.hstack([w.flatten(), b.flatten()]) for w, b in zip(dW, db)])

    def fit(self, X, y):
        # suppose we get data as rows -> transfer them into columns for standard form of mtx calculations
        X = X.T

        # set correct numbers of neurons in first and last layer
        self.layer_sizes[0] = X.shape[0]
        if self.layer_sizes[-1] is None:
            self.layer_sizes[-1] = max(y) + 1

        # initializing bias and weights
        self.bias = [np.random.randn(y, 1) for y in self.layer_sizes[1:]]
        self.weight = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

        # optimization using fmin_l_bfgs_b
        initial_vec = np.hstack([np.hstack([w.flatten(), b.flatten()]) for w, b in zip(self.weight, self.bias)])

        new_vec, cost_min, info = fmin_l_bfgs_b(func=self.cost_for_optimization, x0=initial_vec,
                                                fprime=self.backprop_for_optimization, args=[X, y])

        self.weight, self.bias = update_network(self, new_vec)

        return self

    def predict(self, X):
        X = X.T
        _, A = forward(self, X)
        return A[-1].T

    def weights(self):
        return [np.hstack([w, b]) for w, b in zip(self.weight, self.bias)]


class ANNRegression:

    def __init__(self, units, lambda_, n_input=None, activation_fn='sigmoid'):
        self.nr_layers = len(units) + 2
        self.layer_sizes = [n_input, *units, 1]
        self.bias = None
        self.weight = None
        self.reg = lambda_
        self.activation = activation_fn
        self.act_last = 'linear'

    def cost_for_optimization(self, weight_bias_vec, X, y):
        # change weights and bias of current net to new ones
        self.weight, self.bias = update_network(self, weight_bias_vec)
        # return cost
        return mse_cost(self, X, y)

    def backprop_for_optimization(self, weight_bias_vec, X, y):
        dW, db = backpropagation(self, X, y)
        return np.hstack([np.hstack([w.flatten(), b.flatten()]) for w, b in zip(dW, db)])

    def fit(self, X, y):
        # suppose we get data as rows -> transfer them into columns for standard form of mtx calculations
        X = X.T

        # set correct numbers of neurons in first layer
        self.layer_sizes[0] = X.shape[0]

        # initializing bias and weights
        self.bias = [np.random.randn(y, 1) for y in self.layer_sizes[1:]]
        self.weight = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

        # optimization using fmin_l_bfgs_b
        initial_vec = np.hstack([np.hstack([w.flatten(), b.flatten()]) for w, b in zip(self.weight, self.bias)])

        new_vec, cost_min, info = fmin_l_bfgs_b(func=self.cost_for_optimization, x0=initial_vec,
                                                fprime=self.backprop_for_optimization, args=[X, y])

        self.weight, self.bias = update_network(self, new_vec)

        return self

    def predict(self, X):
        X = X.T
        _, A = forward(self, X)
        return A[-1][0, :]

    def weights(self):
        return [np.hstack([w, b]).T for w, b in zip(self.weight, self.bias)]


def check_gradient(net, x, y, h=1e-6, tol=1e-4):
    dW, db = backpropagation(net, x, y)

    f = cost(net, x, y)

    all_grad_same = True

    for l in range(len(db)):
        I, J = dW[l].shape
        for i in range(I):
            for j in range(J):
                net.weight[l][i, j] += h
                num_dw = (cost(net, x, y) - f) / h
                grad_same = abs(num_dw - dW[l][i, j]) < tol
                if not grad_same:
                    print(f'Gradients are not the same on layer {l+1}, weight {i, j}')
                    print(f'Backprop: {dW[l][i, j]}')
                    print(f'numerical: {num_dw}')
                    all_grad_same = False
                net.weight[l][i, j] -= h
            net.bias[l][i] += h
            num_db = (cost(net, x, y) - f) / h
            grad_same = abs(db[l][i] - num_db) < tol
            if not grad_same:
                print(f'Gradients are not the same on layer {l+1}, bias {i}')
                print(f'Backprop: {db[l][i]}')
                print(f'numerical: {num_db}')
                all_grad_same = False
            net.bias[l][i] -= h
    if all_grad_same:
        print('Congratulations Urša, all partial derivatives are finally the same!!')


if __name__ == '__main__':
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 2, 3])
    net = ANNRegression([1, 3], 2, 4, activation_fn='relu')
    net.fit(X, y)
    a, b = forward(net, np.array([[1, 2]]).T)
    out = b[-1]
    dw, db = backpropagation(net, np.array([[1, 2]]).T, [1])
    # f = cross_entropy_cost(net, np.array([[1, 2]]).T, [1])
    # f1 = cross_entropy_cost(net, np.array([[1, 2]]).T, [0])
    check_gradient(net, X.T, y)
    c = 0

