import random

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def random_feature_forest(X, rand):
    return rand.sample(range(X.shape[1]), int(round(np.sqrt(X.shape[1]))))


def all_features(X, rand):
    return range(X.shape[1])


class Tree:

    def __init__(self, rand=random.Random(1), get_candidate_columns=all_features, min_samples=2):
        self.left = None
        self.right = None
        self.prediction = None
        self.split_criteria = (None, None)
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples

    def build(self, X, y):

        if len(y) < self.min_samples or len(np.unique(y)) == 1:             # stopping criteria
            c, nr = np.unique(y, return_counts=True)
            if len(nr) > 1 and nr[0] == nr[1]:
                self.prediction = self.rand.choice(c)
            else:
                self.prediction = c[np.argmax(nr)]

        else:                                                               # finding the best split
            best_q = np.inf
            best_x = None
            best_s = None
            idx = np.argsort(X, 0)
            for i in self.get_candidate_columns(X, self.rand):
                for s in range(1, idx.shape[0]):
                    if X[idx[s-1, i], i] != X[idx[s, i], i]:                # can't split in between the same values
                        y_l = y[idx[:s, i]]
                        y_r = y[idx[s:, i]]
                        q = len(y_l) * gini(y_l) + len(y_r) * gini(y_r)
                        if q < best_q:
                            best_x = i
                            best_s = s
                            best_q = q

            if best_x is None:                                              # all rows in candidate columns are the same
                c, nr = np.unique(y, return_counts=True)
                if len(nr) > 1 and nr[0] == nr[1]:
                    self.prediction = self.rand.choice(c)
                else:
                    self.prediction = c[np.argmax(nr)]
                return self

            best_split = [idx[:best_s, best_x], idx[best_s:, best_x]]
            self.split_criteria = (best_x, (X[idx[best_s-1, best_x], best_x] + X[idx[best_s, best_x], best_x]) / 2)
            self.left = Tree(rand=self.rand, get_candidate_columns=self.get_candidate_columns, min_samples=self.min_samples)
            self.right = Tree(rand=self.rand, get_candidate_columns=self.get_candidate_columns, min_samples=self.min_samples)

            self.left = self.left.build(X[best_split[0], :], y[best_split[0]])
            self.right = self.right.build(X[best_split[1], :], y[best_split[1]])

        return self

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            x = X[i, :]
            predictions.append(self.predict_one(x))
        return predictions

    def predict_one(self, x):
        if self.prediction is None:
            if x[self.split_criteria[0]] <= self.split_criteria[1]:
                return self.left.predict_one(x)
            else:
                return self.right.predict_one(x)
        else:
            return self.prediction


class Bagging:

    def __init__(self, rand=random.Random(1), tree_builder=Tree(), n=50):
        self.rand = rand
        self.tree_builder = tree_builder
        self.n = n
        self.tree_list = []

    def build(self, X, y):
        for i in range(self.n):
            idx = self.rand.choices([i for i in range(len(y))], k=len(y))
            new_tree = Tree(self.tree_builder.rand, self.tree_builder.get_candidate_columns, self.tree_builder.min_samples)
            self.tree_list.append(new_tree.build(X[idx], y[idx]))
        return self

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(self.predict_one(X[i, :]))
        return predictions

    def predict_one(self, x):
        predictions = []
        for i in range(self.n):
            predictions.append(self.tree_list[i].predict_one(x))

        c, nr = np.unique(predictions, return_counts=True)
        if len(nr) > 1 and nr[0] == nr[1]:                  # if both classes are equally possible, choose random
            prediction = self.rand.choice(c)
        else:
            prediction = c[np.argmax(nr)]
        return prediction


class RandomForest:

    def __init__(self, rand=random.Random(1), n=50, min_samples=2):
        self.rand = rand
        self.tree_builder = Tree(rand, random_feature_forest, min_samples)
        self.n = n
        self.bag = Bagging(self.rand, self.tree_builder, self.n)

    def build(self, X, y):
        return self.bag.build(X, y)

    def predict(self, X):
        return self.bag.predict(X)


def gini(data_y):
    c, nr = np.unique(data_y, return_counts=True)
    g = 1
    for m in nr:
        g -= (m / len(data_y)) ** 2
    return g


def hw_tree_full(train, test):
    X_train, y_train = train
    X_test, y_test = test

    tree = Tree(min_samples=2)
    tree.build(X_train, y_train)

    predictions_train = tree.predict(X_train)
    predictions_test = tree.predict(X_test)

    train_miss = sum(predictions_train != y_train) / len(y_train)
    test_miss = sum(predictions_test != y_test) / len(y_test)

    return train_miss, test_miss


def hw_cv_min_samples(train, test):
    X_train, y_train = train
    X_test, y_test = test

    # shuffling data
    rand = random.Random(0)
    index_shuffle = list(range(X_train.shape[0]))
    rand.shuffle(index_shuffle)

    # cross validation
    possible_min_samples = range(2, 51)
    cv_results_test = np.zeros([5, len(possible_min_samples)])
    cv_results_train = np.zeros([5, len(possible_min_samples)])

    for i in range(5):
        fold_test_idx = index_shuffle[round(i*0.2*len(y_train)):round((i+1)*0.2*len(y_train))]
        fold_train_idx = index_shuffle[:round(i*0.2*len(y_train))] + index_shuffle[round((i+1)*0.2*len(y_train)):]

        for ms in range(len(possible_min_samples)):
            fold_tree = Tree(min_samples=possible_min_samples[ms])
            fold_tree.build(X_train[fold_train_idx, :], y_train[fold_train_idx])

            predictions_test = fold_tree.predict(X_train[fold_test_idx, :])
            predictions_train = fold_tree.predict(X_train[fold_train_idx, :])

            if len(fold_test_idx) != 0:
                cv_results_test[i, ms] = sum(predictions_test != y_train[fold_test_idx]) / len(fold_test_idx)
                cv_results_train[i, ms] = sum(predictions_train != y_train[fold_train_idx]) / len(fold_train_idx)
            else:       # if there exists a fold with no data
                cv_results_test[i, ms] = None
                cv_results_train[i, ms] = None

    # averaging results from cross validation
    mean_error_test = np.mean(cv_results_test, 0)
    mean_error_train = np.mean(cv_results_train, 0)
    best_min_samp = possible_min_samples[mean_error_test.argmin()]

    # plotting
    plt.plot(possible_min_samples, mean_error_train, label='Train')
    plt.plot(possible_min_samples, mean_error_test, label='Test')
    plt.vlines(best_min_samp, 0, 0.2, linestyles='dashed', label='Best value', colors='black')
    plt.title('Misclassification rates in cross validation')
    plt.xlabel('min_samples')
    plt.legend()
    plt.savefig('min_samples.png')
    plt.show()

    # building a tree with the best min_samples value from 5-fold cross validation
    tree = Tree(min_samples=best_min_samp)
    tree.build(X_train, y_train)

    predictions_train = tree.predict(X_train)
    predictions_test = tree.predict(X_test)

    train_miss = sum(predictions_train != y_train) / len(y_train)
    test_miss = sum(predictions_test != y_test) / len(y_test)

    return train_miss, test_miss, best_min_samp


def hw_bagging(train, test, n=50, seed=1):
    X_train, y_train = train
    X_test, y_test = test

    t = Tree(min_samples=2, rand=random.Random(seed))

    b = Bagging(n=n, tree_builder=t)
    bag = b.build(X_train, y_train)

    pred_train = bag.predict(X_train)
    pred_test = bag.predict(X_test)

    train_miss = sum(pred_train != y_train) / len(y_train)
    test_miss = sum(pred_test != y_test) / len(y_test)

    return train_miss, test_miss


def hw_randomforests(train, test, n=50, seed=1):
    X_train, y_train = train
    X_test, y_test = test

    f = RandomForest(n=n, min_samples=2, rand=random.Random(seed))
    forest = f.build(X_train, y_train)

    pred_train = forest.predict(X_train)
    pred_test = forest.predict(X_test)

    train_miss = sum(pred_train != y_train) / len(y_train)
    test_miss = sum(pred_test != y_test) / len(y_test)

    return train_miss, test_miss


def bag_rf_plots(train, test):
    possible_n = [1, 2, 3, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]
    # possible_n = list(range(1, 50))
    seeds = list(range(3))
    colors = ['royalblue', 'orange', 'darkred']
    results = np.zeros([2, 2, len(possible_n), len(seeds)])     # bagging/rf, train/test, n, seeds

    for i in range(len(possible_n)):
        for j in range(len(seeds)):
            results[0, :, i, j] = hw_bagging(train, test, possible_n[i], seeds[j])
            results[1, :, i, j] = hw_randomforests(train, test, possible_n[i], seeds[j])

    # plot for bagging
    for j in range(len(seeds)):
        plt.plot(possible_n, results[0, 0, :, j], label='Train', linestyle='--', color=colors[j])
        plt.plot(possible_n, results[0, 1, :, j], label='Test', linestyle='-', color=colors[j])
    plt.title('Misclassification rates at bagging')
    plt.xlabel('Number of trees')
    plt.legend(labels=['Test', 'Train'], title='Line style')
    plt.savefig('bag_100.png')
    plt.show()

    # plot for random forest
    for j in range(len(seeds)):
        plt.plot(possible_n, results[1, 0, :, j], label='Train', linestyle='--', color=colors[j])
        plt.plot(possible_n, results[1, 1, :, j], label='Test', linestyle='-', color=colors[j])

    plt.title('Misclassification rates at random forests')
    plt.xlabel('Number of trees')
    plt.legend(labels=['Test', 'Train'], title='Line style')
    plt.savefig('rf_100.png')
    plt.show()


if __name__ == "__main__":
    random.seed(1)
    data = pd.read_csv('housing3.csv')
    n = data.shape[0]
    train0 = data.iloc[:round(0.8 * n), :].values
    train = train0[:, :-1], train0[:, -1]
    test0 = data.iloc[round(0.8 * n):, :].values
    test = test0[:, :-1], test0[:, -1]

    miss_train, miss_test = hw_tree_full(train, test)
    print(f'Train misclassification for min_samples=2: {miss_train}')
    print(f'Test misclassification for min_samples=2: {miss_test}')

    miss_train_cv, miss_test_cv, best = hw_cv_min_samples(train, test)
    print(f'Train misclassification for min_samples={best}: {miss_train_cv}')
    print(f'Test misclassification for min_samples={best}: {miss_test_cv}')

    miss_train_bag, miss_test_bag = hw_bagging(train, test)
    print(f'Train misclassification for bagging: {miss_train_bag}')
    print(f'Test misclassification for bagging: {miss_test_bag}')

    miss_train_for, miss_test_for = hw_randomforests(train, test)
    print(f'Train misclassification for random forest: {miss_train_for}')
    print(f'Test misclassification for random forest: {miss_test_for}')

    bag_rf_plots(train, test)

