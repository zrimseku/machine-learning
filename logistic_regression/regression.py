import numpy as np
import scipy.optimize
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def log_likelihood_multi(beta, X, y):
    l = 0
    B_temp = beta.reshape([max(y), X.shape[1] + 1])
    b0 = B_temp[:, 0].reshape([max(y), 1])
    B = B_temp[:, 1:]
    for i in range(X.shape[0]):
        x = np.array([X[i, :]]).T
        c = y[i]
        u = np.vstack((np.matmul(B, x) + b0, np.array([[0]])))
        softmax = np.exp(u)
        softmax /= sum(softmax)
        l -= np.log(softmax[c])
    return l


def inv_logit(x):
    return 1/(1 + np.exp(-x))


def log_likelihood_ord(params, X, y):
    b0, beta, delta = params[0], params[1:X.shape[1]+1], params[X.shape[1]+1:]
    t = np.hstack((np.array([-np.inf, 0]), np.cumsum(delta), np.array([np.inf])))
    u = np.matmul(X, beta) + b0
    p = inv_logit(t[y+1] - u) - inv_logit(t[y] - u)
    p = np.log(p)
    return - sum(p)


class MultinomialLogReg:

    def __init__(self, beta0=None):
        self.beta = beta0

    def build(self, X, y):
        if self.beta is None:
            self.beta = np.ones(max(y) * (X.shape[1] + 1))

        self.beta, _, _ = scipy.optimize.fmin_l_bfgs_b(log_likelihood_multi, self.beta, args=(X, y), approx_grad=True)
        self.beta = self.beta.reshape([max(y), X.shape[1] + 1])

        return self

    def predict(self, X, return_prob=False):
        if return_prob:
            y = np.zeros([X.shape[0], self.beta.shape[0] + 1])
        else:
            y = []
        for i in range(X.shape[0]):
            x = np.array([X[i, :]]).T
            u = np.vstack((np.matmul(self.beta[:, 1:], x) + self.beta[:, 0].reshape([self.beta.shape[0], 1]),
                           np.array([[0]])))
            softmax = np.exp(u)
            softmax /= sum(softmax)
            if return_prob:
                y[i, :] = softmax.T[0]
            else:
                y.append(np.argmax(softmax))
        return y


class OrdinalLogReg:

    def __init__(self, beta=None, delta=None):
        self.beta = beta
        self.delta = delta

    def build(self, X, y):
        if self.beta is None:
            self.beta = np.ones(X.shape[1] + 1) / 2
        if self.delta is None:
            self.delta = np.ones(max(y)-1)
        params, _, _ = scipy.optimize.fmin_l_bfgs_b(log_likelihood_ord, np.hstack((self.beta, self.delta)), args=(X, y),
                                                    approx_grad=True,
                                                    bounds=list((None, None) for _ in range(len(self.beta))) +
                                                           list((1e-10, None) for _ in range(len(self.delta))))
        self.beta, self.delta = params[:X.shape[1]+1], params[X.shape[1]+1:]
        return self

    def predict(self, X, return_prob=False):
        if return_prob:
            y = np.zeros([X.shape[0], len(self.delta)+2])
        else:
            y = []
        t = np.hstack((np.array([-np.inf, 0]), np.cumsum(self.delta), np.array([np.inf])))
        for i in range(X.shape[0]):
            x = np.array([X[i, :]]).T
            u = np.dot(self.beta[1:], x) + self.beta[0]
            c = []
            for j in range(len(t)-1):
                c.append(inv_logit(t[j + 1] - u) - inv_logit(t[j] - u))
            if return_prob:
                y[i, :] = c
            else:
                y.append(np.argmax(c))
        return y


def preprocess(data):
    responses = {'very poor': 0, 'poor': 1, 'average': 2, 'good': 3, 'very good': 4}
    sexes = {'M': 0, 'F': 1}
    data = data.replace({'sex': sexes, 'response': responses})
    y = data['response'].values
    X = data.drop('response', axis=1).values
    return X, y


def test_80_20(X, y):
    t = round(len(y) * 0.8)
    X_train = X[:t, :]
    X_test = X[t:, :]

    y_train = y[:t]
    y_test = y[t:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    M = MultinomialLogReg()
    M.build(X_train, y_train)
    y_pred_m = M.predict(X_test)

    O = OrdinalLogReg()
    O.build(X_train, y_train)
    y_pred_o = O.predict(X_test)

    acc_multi = sum(y_pred_m == y_test)/len(y_test)
    acc_ord = sum(y_pred_o == y_test)/len(y_test)

    return acc_multi, acc_ord


def predict_naive(n, p=(.15, .1, .05, .4, .3)):
    return np.random.choice(5, size=n, p=p)


def cross_validation(X, y, k=5):
    log_loss_multi = np.zeros(k)
    log_loss_ord = np.zeros(k)
    log_loss_naive = np.zeros(k)

    shf_idx = np.arange(X.shape[0])
    np.random.shuffle(shf_idx)

    for f in range(k):
        multi = MultinomialLogReg()
        ordinal = OrdinalLogReg()

        idx_start = round(f*len(shf_idx)/k)
        idx_end = round((f+1)*len(shf_idx)/k)
        test_idx = shf_idx[idx_start:idx_end]
        train_idx = [idx for idx in shf_idx if idx not in test_idx]

        X_test, y_test = X[test_idx, :], y[test_idx]
        X_train, y_train = X[train_idx, :], y[train_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        multi.build(X_train, y_train)
        ordinal.build(X_train, y_train)

        prob_multi = multi.predict(X_test, return_prob=True)
        prob_ord = ordinal.predict(X_test, return_prob=True)
        prob_naive = np.array([[.15, .1, .05, .4, .3], ]*len(y_test))

        for i in range(len(y_test)):
            log_loss_multi[f] -= np.log(prob_multi[i, y_test[i]])
            log_loss_ord[f] -= np.log(prob_ord[i, y_test[i]])
            log_loss_naive[f] -= np.log(prob_naive[i, y_test[i]])

    log_loss_multi /= len(y_test)
    log_loss_ord /= len(y_test)
    log_loss_naive /= len(y_test)

    return log_loss_multi, log_loss_ord, log_loss_naive


def create_dataset(train_size=10):
    size = train_size + 1000
    np.random.seed(0)
    Y = np.random.randint(0, 5, size)
    x1 = lambda y: 2**y
    x2 = lambda y: 3*y**3 + (y-3)**3
    x3 = lambda y: -y + (y-1)**2 - y**3 + np.exp(y)
    x4 = lambda y: np.exp(y/5 - y**2 + (y-3)**3/23)

    X1 = x1(Y) + np.random.uniform(-1, 1, size)
    X2 = x2(Y) + np.random.uniform(-1, 1, size)
    X3 = x3(Y) + np.random.uniform(-1, 1, size)
    X4 = x4(Y) + np.random.uniform(-1, 1, size)

    dataset = pd.DataFrame({'response': Y, 'x1': X1, 'x2': X2, 'x3': X3, 'x4': X4})

    names = {0: 'very poor', 1: 'poor', 2: 'average', 3: 'good', 4: 'very good'}
    dataset = dataset.replace({'response': names})
    dataset[:train_size].to_csv('multinomial_bad_ordinal_good_train.csv', index=False, sep=';', decimal=',')
    dataset[train_size:].to_csv('multinomial_bad_ordinal_good_test.csv', index=False, sep=';', decimal=',')


def test_new():
    train = pd.read_csv('multinomial_bad_ordinal_good_train.csv', sep=';', thousands=',')
    test = pd.read_csv('multinomial_bad_ordinal_good_test.csv', sep=';', thousands=',')

    responses = {'very poor': 0, 'poor': 1, 'average': 2, 'good': 3, 'very good': 4}
    train = train.replace({'response': responses})
    test = test.replace({'response': responses})
    y_train = train['response'].values
    X_train = train.drop('response', axis=1).values
    y_test = test['response'].values
    X_test = test.drop('response', axis=1).values

    log_loss_multi = 0
    log_loss_ord = 0
    log_loss_naive = 0

    classes_mult = 0
    classes_ord = 0

    multi = MultinomialLogReg()
    ordinal = OrdinalLogReg()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    multi.build(X_train, y_train)
    ordinal.build(X_train, y_train)

    prob_multi = multi.predict(X_test, return_prob=True)
    prob_ord = ordinal.predict(X_test, return_prob=True)
    prob_naive = np.array([[.15, .1, .05, .4, .3], ]*len(y_test))

    c_multi = multi.predict(X_test)
    c_ord = ordinal.predict(X_test)

    for i in range(len(y_test)):
        log_loss_multi -= np.log(prob_multi[i, y_test[i]])
        log_loss_ord -= np.log(prob_ord[i, y_test[i]])
        log_loss_naive -= np.log(prob_naive[i, y_test[i]])

        if c_multi[i] == y_test[i]:
            classes_mult += 1
        if c_ord[i] == y_test[i]:
            classes_ord += 1

    log_loss_multi /= len(y_test)
    log_loss_ord /= len(y_test)
    log_loss_naive /= len(y_test)
    classes_mult /= len(y_test)
    classes_ord /= len(y_test)
    classes_naive = sum(y_test == 3) / len(y_test)

    return log_loss_multi, log_loss_ord, log_loss_naive, classes_mult, classes_ord, classes_naive


def bootstrap_coeff(X, y, repetitions=100):
    betas = np.zeros([repetitions, X.shape[1] + 1])
    for i in range(repetitions):
        idx = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace=True)
        X_boot = X[idx, :]
        y_boot = y[idx]

        scaler = StandardScaler()
        X_boot = scaler.fit_transform(X_boot)

        O = OrdinalLogReg()
        O.build(X_boot, y_boot)
        betas[i, :] = O.beta

    mean = betas.mean(axis=0)
    betas.sort(axis=0)
    percl = betas[round(2.5 * repetitions / 100), :]
    perch = betas[round(97.5 * repetitions / 100), :]

    return mean, percl, perch


if __name__ == "__main__":
    # 80/20
    data = pd.read_csv('dataset.csv', sep=';')
    X, y = preprocess(data)
    p1, p2 = test_80_20(X, y)
    print(f'Accuracy on 80/20 split:'
          f'    - multinomial: {p1}, '
          f'    - ordinal: {p2}')

    # CROSS VALIDATION
    np.random.seed(0)
    llm, llo, lln = cross_validation(X, y, 10)
    print('CROSS VALIDATION')
    print(f'Mean of log loss for:'
          f'    - Multinomial: {np.mean(llm)}'
          f'    - Ordinal: {np.mean(llo)}'
          f'    - Naive: {np.mean(lln)}')
    print(f'Standard deviation of log loss for:'
          f'    - Multinomial: {np.std(llm)}'
          f'    - Ordinal: {np.std(llo)}'
          f'    - Naive: {np.std(lln)}')

    # NEW DATASET
    create_dataset(10)
    llm, llo, lln, p1, p2, p3 = test_new()
    print('NEW DATASET')
    print(f'Accuracy on 1000 test points of the new dataset:'
          f'    - multinomial: {p1}, '
          f'    - ordinal: {p2},'
          f'    - naive: {p3}')

    print(f'Mean log loss on 1000 test points of the new dataset:'
          f'    - multinomial: {llm}, '
          f'    - ordinal: {llo},'
          f'    - naive: {lln}')

    # BOOTSTRAP - coefficient analysis
    data = pd.read_csv('dataset.csv', sep=';')

    X, y = preprocess(data)

    beta, pl, ph = bootstrap_coeff(X, y, 1000)

    names = data.columns.values[1:]
    names = np.hstack((np.array(['bias']), names))
    ci = np.array([beta - pl, ph - beta])
    plt.bar(names, beta, yerr=ci)
    plt.savefig('bootstrap1000.png')
    plt.show()

    # CORRELATION
    data = pd.read_csv('dataset.csv', sep=';')
    responses = {'very poor': 0, 'poor': 1, 'average': 2, 'good': 3, 'very good': 4}
    sexes = {'M': 0, 'F': 1}
    D = data.replace({'sex': sexes, 'response': responses})

    corrmat = D.corr()
    f, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corrmat, annot=True, fmt='.1g', vmin=-1, vmax=1, cmap='RdBu')
    plt.show()
