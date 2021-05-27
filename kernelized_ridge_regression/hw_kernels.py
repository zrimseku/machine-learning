import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Kernel:

    def __call__(self, A, B):
        pass


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

        pred = model.predict(X_test_cv).T

        r = np.sqrt(np.sum((pred - y_test_cv) ** 2) / len(y_test_cv))

        rmse[f] = r

    return np.mean(rmse)


if __name__ == '__main__':
    # SINE DATA
    # comparison of parameters for both methods
    sine_plot(0.1)

    # parameters that look best:
    lmb = 0.1
    m = 15
    sigma = 0.2

    # comparison of polynomial an rbf kernel
    x, y, sine_mean, sine_std = read_sine()

    reg_poly = KernelizedRidgeRegression(Polynomial(m), lmb)
    reg_rbf = KernelizedRidgeRegression(RBF(sigma), lmb)
    reg_poly = reg_poly.fit(x, y)
    reg_rbf = reg_rbf.fit(x, y)

    plt.figure()
    plt.scatter(x * sine_std + sine_mean, y, color='gray')

    seq = np.array([np.linspace(0, 20, 100)])
    seq = (seq - sine_mean) / sine_std

    predicted_pol = reg_poly.predict(seq.T)
    plt.plot(seq[0] * sine_std + sine_mean, predicted_pol[0], label='polynomial')
    predicted_rbf = reg_rbf.predict(seq.T)
    plt.plot(seq[0] * sine_std + sine_mean, predicted_rbf[0], label='rbf')

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

    # shuffling index for cross validation - we want to have the same data for testing different parameters
    np.random.seed(1)
    shf_idx = np.arange(X_train.shape[0])
    np.random.shuffle(shf_idx)

    M = range(1, 11)
    sigma = np.linspace(0.01, 20, 2000)
    L = [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]

    RMSE_rbf = np.zeros([2, len(sigma)])
    RMSE_poly = np.zeros([2, len(M)])
    best_lambdas_rbf = np.zeros(len(sigma))
    best_lambdas_poly = np.zeros(len(M))

    # calculate RMSE_poly
    for i in range(len(M)):
        m = M[i]
        best_lambda = -1
        min_rmse = np.infty
        for l in L:
            model = KernelizedRidgeRegression(Polynomial(m), l)
            rmse = cross_validation(model, shf_idx, X_train, y_train)
            if rmse < min_rmse:
                best_lambda = l
                min_rmse = rmse

        best_lambdas_poly[i] = best_lambda
        models = [KernelizedRidgeRegression(Polynomial(m), 1), KernelizedRidgeRegression(Polynomial(m), best_lambda)]

        for j in range(2):
            model = models[j]
            model = model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test).T
            RMSE_poly[j, i] = np.sqrt(np.sum((pred - y_test) ** 2) / len(y_test))

    # calculate RMSE_rbf
    for i in range(len(sigma)):
        sgm = sigma[i]
        best_lambda = -1
        min_rmse = np.infty
        for l in L:
            model = KernelizedRidgeRegression(RBF(sgm), l)
            rmse = cross_validation(model, shf_idx, X_train, y_train)
            if rmse < min_rmse:
                best_lambda = l
                min_rmse = rmse

        best_lambdas_rbf[i] = best_lambda
        models = [KernelizedRidgeRegression(RBF(sgm), 1), KernelizedRidgeRegression(RBF(sgm), best_lambda)]

        for j in range(2):
            model = models[j]
            model = model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test).T
            RMSE_rbf[j, i] = np.sqrt(np.sum((pred - y_test) ** 2) / len(y_test))

    print(f'Best lambdas rbf (::100): {best_lambdas_rbf[::100]}')
    print(f'Best lambdas polynomial: {best_lambdas_poly}')

    # Plotting
    plt.figure()
    plt.plot(M, RMSE_poly[0, :], label='1')
    plt.plot(M, RMSE_poly[1, :], label='best')
    plt.legend(title='lambda:')
    plt.show()

    print(RMSE_poly)

    plt.figure()
    plt.plot(sigma, RMSE_rbf[0, :], label='1')
    plt.plot(sigma, RMSE_rbf[1, :], label='best')
    plt.legend(title='lambda:')
    plt.show()

    best1 = np.argmin(RMSE_rbf[0, :])
    best = np.argmin(RMSE_rbf[1, :])

    print(f'Sigma value for min RMSE: '
          f'    - lambda = 1: {sigma[best1]}'
          f'    - lambda = {best_lambdas_rbf[best]}: {sigma[best]}')