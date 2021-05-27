import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix


ERROR = 1e-06


class Kernel:

    def __call__(self, A, B):
        pass


class SVR:

    def __init__(self, kernel: Kernel, lambda_, epsilon=1e-5):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.X = None
        self.alpha = None
        self.alpha_star = None
        self.b = None
        self.vectors = None

    def fit(self, X, y):
        self.X = X

        if len(y.shape) == 1:
            y = np.array([y]).T

        K = self.kernel(X, X)
        sh = K.shape
        coeff = np.tile(np.array([[1, -1], [-1, 1]]), sh)
        P = np.repeat(np.repeat(K, 2, 0), 2, 1) * coeff

        A = np.tile(np.array([[1, -1]]), sh[0])
        b = np.zeros(1)

        q = self.epsilon - np.repeat(y, 2, 0) * A.T

        h = np.zeros((sh[0]*4, 1))
        C = 1 / self.lambda_
        h[sh[0]*2:] = C
        G = np.vstack([-np.eye(sh[0] * 2), np.eye(sh[0] * 2)])

        solvers.options['show_progress'] = False
        result = solvers.qp(matrix(P, tc='d'), matrix(q, tc='d'), matrix(G, tc='d'), matrix(h, tc='d'),
                            matrix(A, tc='d'), matrix(b, tc='d'))
        result = result['x']

        self.alpha = np.array(result[::2])
        self.alpha_star = np.array(result[1::2])

        W = K @ (self.alpha - self.alpha_star)

        b_lower = y - self.epsilon - W
        b_upper = y + self.epsilon - W

        low_valid = np.logical_or(self.alpha < C - ERROR, self.alpha_star > ERROR)
        upp_valid = np.logical_or(self.alpha_star < C - ERROR, self.alpha > ERROR)

        b_max = np.max(b_lower[low_valid])
        b_min = np.min(b_upper[upp_valid])

        self.b = (b_max + b_min) / 2

        vector_idx = np.squeeze(np.abs(self.alpha - self.alpha_star) > ERROR)
        self.vectors = X[vector_idx, :]

        return self

    def predict(self, X_new):
        k = self.kernel(self.X, X_new)
        return np.squeeze((self.alpha - self.alpha_star).T.dot(k) + self.b)

    def get_alpha(self):
        return np.hstack([self.alpha, self.alpha_star])

    def get_b(self):
        return self.b


class KernelizedRidgeRegression:

    def __init__(self, kernel: Kernel, lambda_):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.X = None
        self.alpha = None

    def fit(self, X, y):
        self.X = X

        Z = self.kernel(X, X) + self.lambda_ * np.eye(len(y))

        self.alpha = np.linalg.inv(Z) @ y

        return self

    def predict(self, X_new):
        return self.alpha.T @ self.kernel(self.X, X_new)


class RBF(Kernel):

    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, A, B):

        if A.ndim == 1:
            A = np.array([A])
        if B.ndim == 1:
            B = np.array([B])

        normA = np.array([np.linalg.norm(A, axis=1)**2])
        normB = np.array([np.linalg.norm(B, axis=1)**2])

        X1 = np.repeat(normA.T, B.shape[0], axis=1)
        X2 = np.repeat(normB, A.shape[0], axis=0)
        C = A @ B.T

        k = np.exp(-1/(2*self.sigma**2) * (X1 + X2 - 2 * C))

        k = np.squeeze(k)
        if k.shape == ():
            k = np.array([k])

        return k


class Polynomial(Kernel):

    def __init__(self, M=2):
        self.M = M

    def __call__(self, A, B):

        if A.ndim == 1:
            A = np.array([A])
        if B.ndim == 1:
            B = np.array([B])

        k = (np.ones([A.shape[0], B.shape[0]]) + A @ B.T)**self.M

        k = np.squeeze(k)
        if k.shape == ():
            k = np.array([k])

        return k


def read_sine():
    sine_data = pd.read_csv('sine.csv')
    x = np.array([sine_data['x'].values]).T
    y = np.array([sine_data['y'].values]).T

    sine_mean = np.mean(x)
    sine_std = np.std(x)

    x = (x - sine_mean) / sine_std

    return x, y, sine_mean, sine_std


def sine_plot(lmb):

    x, y, sine_mean, sine_std = read_sine()

    plt.figure()
    plt.scatter(x * sine_std + sine_mean, y, color='gray')
    # for m in [10, 20, 30, 50]:
    for m in [1, 2, 3, 5, 8, 10, 15, 20]:
        kernel_pol = Polynomial(m)
        reg_pol = KernelizedRidgeRegression(kernel_pol, lmb)

        fitted = reg_pol.fit(x, y)

        seq = np.array([np.linspace(0, 20, 100)])
        seq = (seq - sine_mean) / sine_std
        predicted = fitted.predict(seq.T)

        plt.plot(seq[0] * sine_std + sine_mean, predicted[0], label=f'{m}')

    plt.legend(loc=2, title='M:')
    plt.show()

    plt.figure()
    plt.scatter(x * sine_std + sine_mean, y, color='gray')
    for sgm in [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2]:
        kernel_rbf = RBF(sgm)
        reg_rbf = KernelizedRidgeRegression(kernel_rbf, lmb)

        fitted = reg_rbf.fit(x, y)

        seq = np.array([np.linspace(0, 20, 100)])
        seq = (seq - sine_mean) / sine_std
        predicted = fitted.predict(seq.T)

        plt.plot(seq[0] * sine_std + sine_mean, predicted[0], label=f'{sgm}')
    plt.legend(loc=2, title='sigma:')
    plt.show()


def cross_validation(model, shf_idx, X, y, k=5):
    rmse = np.zeros(k)

    for f in range(k):

        idx_start = round(f*len(shf_idx)/k)
        idx_end = round((f+1)*len(shf_idx)/k)
        test_idx = shf_idx[idx_start:idx_end]
        train_idx = [idx for idx in shf_idx if idx not in test_idx]

        X_test_cv, y_test_cv = X[test_idx, :], y[test_idx]
        X_train_cv, y_train_cv = X[train_idx, :], y[train_idx]

        mean = np.mean(X_train_cv, axis=0)
        std = np.std(X_train_cv, axis=0)

        X_train_cv = (X_train_cv - mean) / std
        X_test_cv = (X_test_cv - mean) / std

        model = model.fit(X_train_cv, y_train_cv)

        pred = np.array([model.predict(X_test_cv)]).T

        r = np.sqrt(np.sum((pred - y_test_cv)**2) / len(y_test_cv))

        rmse[f] = r

    return np.mean(rmse)


if __name__ == '__main__':
    # ERROR = 1e-06

    # SINE DATA
    M = 15
    sigma = 0.2
    x, y, sine_mean, sine_std = read_sine()

    model_poly = SVR(kernel=Polynomial(M), lambda_=0.1, epsilon=0.5)
    model_rbf = SVR(kernel=RBF(sigma), lambda_=0.1, epsilon=0.5)
    fitted_poly = model_poly.fit(x, y)
    fitted_rbf = model_rbf.fit(x, y)

    alpha_poly = fitted_poly.get_alpha()
    vec_idx_poly = np.abs(alpha_poly[:, 0] - alpha_poly[:, 1]) > ERROR

    alpha_rbf = fitted_rbf.get_alpha()
    vec_idx_rbf = np.abs(alpha_rbf[:, 0] - alpha_rbf[:, 1]) > ERROR

    plt.figure()

    plt.scatter(x * sine_std + sine_mean, y, color='gray', alpha=0.2)

    plt.scatter(x[vec_idx_poly] * sine_std + sine_mean, y[vec_idx_poly], color='b', alpha=0.5,
                label='Support vectors Polynomial')
    plt.scatter(x[vec_idx_rbf] * sine_std + sine_mean, y[vec_idx_rbf], color='orange', alpha=0.5,
                label='Support vectors RBF')

    seq = np.array([np.linspace(0, 20, 100)])
    seq = (seq - sine_mean) / sine_std
    predicted_poly = fitted_poly.predict(seq.T)
    predicted_rbf = fitted_rbf.predict(seq.T)

    plt.plot(seq[0] * sine_std + sine_mean, predicted_poly, color='b', label=f'Polynomial, M={M}')
    plt.plot(seq[0] * sine_std + sine_mean, predicted_rbf, color='orange', label=f'RBF, sigma={sigma}')

    plt.legend()
    plt.show()

    # HOUSING DATA
    housing_data = pd.read_csv('housing2r.csv')
    X = housing_data.loc[:, housing_data.columns != 'y'].values
    Y = np.array([housing_data['y'].values]).T

    t = round(len(Y) * 0.8)

    X_train = X[:t, :]
    X_test = X[t:, :]
    y_train = Y[:t]
    y_test = Y[t:]

    housing_mean = np.mean(X_train, axis=0)
    housing_std = np.std(X_train, axis=0)

    X_train_scaled = (X_train - housing_mean) / housing_std
    X_test = (X_test - housing_mean) / housing_std

    model = SVR(Polynomial(10), 0.1, 4)
    model.fit(X_train_scaled, y_train)

    # shuffling index for cross validation - we want to have the same data for testing different parameters
    np.random.seed(42)
    shf_idx = np.arange(X_train.shape[0])
    np.random.shuffle(shf_idx)

    M = range(1, 11)
    sigma = np.linspace(0.01, 20, 100)
    L = np.array([0.1, 0.25, 0.5, 1, 2.5, 5, 10])
    epsilon = 8

    RMSE_rbf = np.zeros([2, len(sigma)])
    RMSE_poly = np.zeros([2, len(M)])
    nr_vec_rbf = np.zeros([2, len(sigma)])
    nr_vec_poly = np.zeros([2, len(M)])
    best_lambdas_rbf = np.zeros(len(sigma))
    best_lambdas_poly = np.zeros(len(M))

    # calculate RMSE_poly
    for i in range(len(M)):
        m = M[i]
        best_lambda = -1
        min_rmse = np.infty
        for l in L:
            model = SVR(Polynomial(m), l, epsilon)
            rmse = cross_validation(model, shf_idx, X_train, y_train)
            if rmse < min_rmse:
                best_lambda = l
                min_rmse = rmse

        best_lambdas_poly[i] = best_lambda
        models = [SVR(Polynomial(m), 1, epsilon), SVR(Polynomial(m), best_lambda, epsilon)]

        for j in range(2):
            model = models[j]
            model = model.fit(X_train_scaled, y_train)
            pred = np.array([model.predict(X_test)]).T
            RMSE_poly[j, i] = np.sqrt(np.sum((pred - y_test) ** 2) / len(y_test))
            nr_vec_poly[j, i] = len(model.vectors)

    # calculate RMSE_rbf
    for i in range(len(sigma)):
        sgm = sigma[i]
        best_lambda = -1
        min_rmse = np.infty
        for l in L:
            model = SVR(RBF(sgm), l, epsilon)
            rmse = cross_validation(model, shf_idx, X_train, y_train)
            if rmse < min_rmse:
                best_lambda = l
                min_rmse = rmse

        best_lambdas_rbf[i] = best_lambda
        models = [SVR(RBF(sgm), 1, epsilon), SVR(RBF(sgm), best_lambda, epsilon)]

        for j in range(2):
            model = models[j]
            model = model.fit(X_train_scaled, y_train)
            pred = np.array([model.predict(X_test)]).T
            RMSE_rbf[j, i] = np.sqrt(np.sum((pred - y_test) ** 2) / len(y_test))
            nr_vec_rbf[j, i] = len(model.vectors)

    print(f'Best lambdas rbf (::10): {best_lambdas_rbf[::10]}')
    print(f'Best lambdas polynomial: {best_lambdas_poly}')

    # Plotting
    plt.figure()
    plt.plot(M, RMSE_poly[0, :], label='1')
    plt.plot(M, RMSE_poly[1, :], label='best')
    plt.legend(title='lambda:')
    plt.show()

    # print(RMSE_poly)

    plt.figure()
    plt.plot(sigma, RMSE_rbf[0, :], label='1')
    plt.plot(sigma, RMSE_rbf[1, :], label='best')
    plt.legend(title='lambda:')
    plt.show()

    # print(RMSE_rbf)

    plt.figure()
    plt.plot(M, nr_vec_poly[0, :], label='1')
    plt.plot(M, nr_vec_poly[1, :], label='best')
    plt.legend(title='lambda:')
    plt.show()

    plt.figure()
    plt.plot(sigma, nr_vec_rbf[0, :], label='1')
    plt.plot(sigma, nr_vec_rbf[1, :], label='best')
    plt.legend(title='lambda:')
    plt.show()

    best1 = np.argmin(RMSE_rbf[0, :])
    best = np.argmin(RMSE_rbf[1, :])

    print(f'Sigma value for min RMSE: '
          f'    - lambda = 1: {sigma[best1]}'
          f'    - lambda = {best_lambdas_rbf[best]}: {sigma[best]}')

