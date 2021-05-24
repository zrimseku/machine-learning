import numpy as np


def cross_entropy_cost(net, x, true_class):
    _, A = forward(net, x)
    y = A[-1]
    t = np.zeros(y.shape[0])
    t[true_class] = 1
    return -np.dot(t, np.log(y))[0] - np.dot(1-t, np.log(1-y))[0]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward(net, x):
    # TODO: za več inputov naenkrat
    Z = [x]
    A = [x]
    for i in range(net.nr_layers - 1):
        z = net.weights[i] @ x + net.bias[i]
        x = sigmoid(z)
        Z.append(z)
        A.append(x)
    return Z, A


def backpropagation(net, x, y):
    Z, A = forward(net, x)
    dW = [np.zeros(np.shape(w)) for w in net.weights]
    db = [np.zeros(np.shape(b)) for b in net.bias]
    dC_a = A[-1]
    dC_a[y] += -1
    db[-1] = dC_a
    dW[-1] = db[-1] @ A[-2].T
    for i in range(2, net.nr_layers):
        db[-i] = A[-i] * (1 - A[-i]) * (net.weights[1-i].T @ db[1-i])
        dW[-i] = db[-i] @ A[-i-1].T
    return dW, db


class ANNClassification:

    def __init__(self, units, n_input=1, n_classes=1, lambda_=0):
        # number of input parameters and classes are set to 1, if we dont know the values at initialization
        self.nr_layers = len(units) + 2
        self.layer_sizes = [n_input, *units, n_classes]
        self.bias = [np.random.randn(y, 1) for y in self.layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]
        self.reg = lambda_


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
    net = ANNClassification([4, 5, 10], 3, 2)
    a, b = forward(net, np.array([[1, 2, 3]]).T)
    out = b[-1]
    dw, db = backpropagation(net, np.array([[1, 2, 3]]).T, 1)
    f = cross_entropy_cost(net, np.array([[1, 2, 3]]).T, 1)
    f1 = cross_entropy_cost(net, np.array([[1, 2, 3]]).T, 0)
    check_gradient(net, np.array([[1, 2, 3]]).T, 1)
    c = 0

