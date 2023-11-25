import numpy as np
import matplotlib.pyplot as plt

from MH import *
from markovchain import *

np.random.seed(99)


def generate_data(d, m, sigma):
    X = np.random.randn(m, d)
    true_theta = np.random.randint(0, 2, size=d)
    noise = sigma * np.random.randn(m)
    y = np.dot(X, true_theta) + noise

    return X, true_theta, y


def estimate_error(d, m, sigma, beta):
    iter = 100 * d

    X, true_theta, y = generate_data(d, m, sigma)

    initial_theta = np.random.randint(0, 2, size=d)

    acceptance_calc = AcceptanceCalculator(X, y, beta)
    chain = BinaryHypercubeChain(acceptance_calc, d, initial_theta)
    mh = MetropolisHastings(chain, iter)

    mh.run()  # compute samples

    estimate_theta = chain.current_state

    return np.linalg.norm(estimate_theta - true_theta) ** 2 * 2 / d

if __name__ == "__main__":
    d = 200
    m = 500 #np.linspace(1000, 10000, 11)
    sigma = 1

    beta = 0.1

    error = []
    for i in range(10):
        error.append(estimate_error(d, m, sigma, beta))

    print(f"Mean Squared Error: {np.mean(error)}")
