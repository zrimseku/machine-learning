import numpy as np


def cross_entropy_cost(net, X, true_classes):
    _, A = forward(net, X)
    Y = A[-1]
    ce = sum(np.log(Y[true_classes, range(len(true_classes))]))
    if net.act_last == 'sigmoid':
        for i in range(len(true_classes)):
            y = A[-1][:, i]
            t = np.zeros(y.shape[0])
            t[true_classes[i]] = 1
            ce += np.dot(1 - t, np.log(1 - y))
    return -ce / len(true_classes)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(net, X):
    Z = [X]
    A = [X]
    for i in range(net.nr_layers - 1):
        z = net.weights[i] @ X + net.bias[i]
        X = sigmoid(z)
        Z.append(z)
        A.append(X)
    return Z, A


def backpropagation(net, X, y):
    Z, A = forward(net, X)
    dW = [np.zeros(np.shape(w)) for w in net.weights]
    db = [np.zeros(np.shape(b)) for b in net.bias]

    dz = A[-1]
    dz[y, range(len(y))] += -1
    db[-1] = np.reshape(np.sum(dz, axis=1), (net.layer_sizes[-1], 1)) / len(y)
    dW[-1] = dz @ A[-2].T / len(y)

    for i in range(2, net.nr_layers):
        dz = A[-i] * (1 - A[-i]) * (net.weights[1-i].T @ dz)
        dW[-i] = dz @ A[-i-1].T / len(y)
        db[-i] = np.reshape(np.sum(dz, axis=1), (dz.shape[0], 1)) / len(y)
    return dW, db


class ANNClassification:

    def __init__(self, units, n_input=1, n_classes=1, lambda_=0, activation_fn='sigmoid', act_last='sigmoid'):
        # number of input parameters and classes are set to 1, if we dont know the values at initialization
        self.nr_layers = len(units) + 2
        self.layer_sizes = [n_input, *units, n_classes]
        self.bias = [np.random.randn(y, 1) for y in self.layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        self.reg = lambda_
        self.activation = activation_fn
        self.act_last = act_last


class ANNRegression:

    def __init__(self, units, lambda_, n_input):
        self.nr_layers = len(units) + 2
        self.layer_sizes = [n_input, *units, 1]
        self.bias = [np.random.randn(y, 1) for y in self.layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        self.reg = lambda_


def check_gradient(net, x, y, h=1e-6, tol=1e-3):
    dW, db = backpropagation(net, x, y)
    f = cross_entropy_cost(net, x, y)
    all_grad_same = True

    for l in range(len(db)):
        I, J = dW[l].shape
        for i in range(I):
            for j in range(J):
                net.weights[l][i, j] += h
                num_dw = (cross_entropy_cost(net, x, y) - f) / h
                grad_same = abs(num_dw - dW[l][i, j]) < tol
                if not grad_same:
                    print(f'Gradients are not the same on layer {l+1}, weight {i, j}')
                    print(f'Backprop: {dW[l][i, j]}')
                    print(f'numerical: {num_dw}')
                    all_grad_same = False
                net.weights[l][i, j] -= h
            net.bias[l][i] += h
            num_db = (cross_entropy_cost(net, x, y) - f) / h
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
    net = ANNClassification([4, 5, 10], 2, 4)
    a, b = forward(net, np.array([[1, 2]]).T)
    out = b[-1]
    dw, db = backpropagation(net, np.array([[1, 2]]).T, [1])
    f = cross_entropy_cost(net, np.array([[1, 2]]).T, [1])
    f1 = cross_entropy_cost(net, np.array([[1, 2]]).T, [0])
    check_gradient(net, X.T, y)
    c = 0

