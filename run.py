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


def estimate_error(chain_type, d, m, sigma, beta, s=None):
    iter = 100 * d

    X, true_theta, y = generate_data(d, m, sigma, s)

    initial_theta = generate_theta(d, s)

    mh = MetropolisHastings(chain_type, d, initial_theta, X, y, beta, iter)
    mh.run()  # compute samples

    estimate_theta = mh.chain.current_state

    return np.linalg.norm(estimate_theta - true_theta) ** 2 * 2 / d


if __name__ == "__main__":
    d = 200
    s = d // 100
    m = 500
    sigma = 1

    beta = 0.1

    q1_error = []
    for i in range(3):
        q1_error.append(estimate_error(BinaryHypercubeChain, d, m, sigma, beta))

    print(f"Mean Squared Error Question 1: {np.mean(q1_error)}")

    q2_error = []
    for i in range(3):
        q2_error.append(estimate_error(SwapChain, d, m, sigma, beta, s))

    print(f"Mean Squared Error Question 2: {np.mean(q2_error)}")
