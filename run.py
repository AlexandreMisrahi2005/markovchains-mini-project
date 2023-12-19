import numpy as np
import matplotlib.pyplot as plt

from MH import *
from markovchain import *

np.random.seed(99)


def generate_theta(d, s=None):
    if s is None:
        theta = np.random.randint(0, 2, size=d)
    else:
        theta = np.zeros(d)
        one_idx = np.random.choice(np.arange(d), size=s, replace=False)
        theta[one_idx] = 1
    return theta


def generate_data(d, m, sigma, s=None):
    X = np.random.randn(m, d)

    true_theta = generate_theta(d, s)

    noise = sigma * np.random.randn(m)
    y = np.dot(X, true_theta) + noise

    return X, true_theta, y


def estimate_error(chain_type, d, m, sigma, beta, s=None, sign_chain=False):
    iter = 100 * d

    X, true_theta, y = generate_data(d, m, sigma, s)

    initial_theta = generate_theta(d, s)

    if sign_chain:
        mh = MetropolisHastings(chain_type, d, initial_theta, X, y, beta, sigma, iter)
    else:
        mh = MetropolisHastings(chain_type, d, initial_theta, X, y, beta, None, iter)
    mh.run()  # compute samples

    estimate_theta = mh.chain.current_state

    return np.linalg.norm(estimate_theta - true_theta) ** 2


def compute_error(estimate_theta, true_theta):
    return np.linalg.norm(estimate_theta - true_theta, axis=-1) ** 2


if __name__ == "__main__":

    question = -1
    # question = 1.6
    # question = 2.3
    # question = 3.2

    d = 128 * 128
    s = 150
    # m = 200
    sigma = 1

    beta = 0.1

    theta = generate_theta(d, s)

    import pandas as pd
    y = pd.read_csv('Measurements_y.csv', sep=',', header=None).values.reshape(-1)
    X =  pd.read_csv('SensingMatrix_X.csv', sep=',', header=None).values
    print(X.shape)

    mh = MetropolisHastings(SkyChain, d, theta, X, y, beta, None, 10 * d)
    mh.run()  # compute samples

    estimate_theta = mh.chain.current_state
    print(estimate_theta)
    print((estimate_theta > 0).sum(), (estimate_theta > 1).sum())
