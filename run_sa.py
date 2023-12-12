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


def estimate_error(chain_type, d, m, sigma, beta, iter, s=None):
    X, true_theta, y = generate_data(d, m, sigma, s)

    initial_theta = generate_theta(d, s)

    mh = MetropolisHastings(chain_type, d, initial_theta, X, y, beta, iter)
    mh.run()  # compute samples

    estimate_theta = mh.chain.current_state

    return np.linalg.norm(estimate_theta - true_theta) ** 2 * 2 / d


def estimate_error_sa(chain_type, d, m, sigma, betas, iter, s=None):
    X, true_theta, y = generate_data(d, m, sigma, s)

    initial_theta = generate_theta(d, s)

    samh = SimulatedAnnealing(chain_type, d, initial_theta, X, y, betas, iter)
    samh.run()  # compute samples

    estimate_theta = samh.chain.current_state

    return np.linalg.norm(estimate_theta - true_theta) ** 2 * 2 / d


if __name__ == "__main__":
    d = 2000
    m = 2000
    sigma = 1

    betas = [10**i for i in range(-2, 4)]

    iter = 100 * d

    for beta in [0.01, 0.1, 1, 10, 100]:
        q1_error = []
        for i in range(5):
            q1_error.append(
                estimate_error(BinaryHypercubeChain, d, m, sigma, beta, iter)
            )
        print(
            f"Mean Squared Error Question 1, beta={beta}, iter={iter}: {np.mean(q1_error)} +- {np.std(q1_error)}"
        )

    q1_error_sa = []
    for i in range(5):
        q1_error_sa.append(
            estimate_error_sa(BinaryHypercubeChain, d, m, sigma, betas, iter)
        )
    print(
        f"Mean Squared Error Question 1 (SA), iter={iter}: {np.mean(q1_error_sa)} +- {np.std(q1_error_sa)}"
    )

    iter = 20 * d

    for beta in [0.01, 0.1, 1, 10, 100]:
        q1_error = []
        for i in range(5):
            q1_error.append(
                estimate_error(BinaryHypercubeChain, d, m, sigma, beta, iter)
            )
        print(
            f"Mean Squared Error Question 1, beta={beta}, iter={iter}: {np.mean(q1_error)} +- {np.std(q1_error)}"
        )

    q1_error_sa = []
    for i in range(5):
        q1_error_sa.append(
            estimate_error_sa(BinaryHypercubeChain, d, m, sigma, betas, iter)
        )
    print(
        f"Mean Squared Error Question 1 (SA), iter={iter}: {np.mean(q1_error_sa)} +- {np.std(q1_error_sa)}"
    )

    iter = 10 * d

    for beta in [0.01, 0.1, 1, 10, 100]:
        q1_error = []
        for i in range(5):
            q1_error.append(
                estimate_error(BinaryHypercubeChain, d, m, sigma, beta, iter)
            )
        print(
            f"Mean Squared Error Question 1, beta={beta}, iter={iter}: {np.mean(q1_error)} +- {np.std(q1_error)}"
        )

    q1_error_sa = []
    for i in range(5):
        q1_error_sa.append(
            estimate_error_sa(BinaryHypercubeChain, d, m, sigma, betas, iter)
        )
    print(
        f"Mean Squared Error Question 1 (SA), iter={iter}: {np.mean(q1_error_sa)} +- {np.std(q1_error_sa)}"
    )

    iter = 5 * d

    for beta in [0.01, 0.1, 1, 10, 100]:
        q1_error = []
        for i in range(5):
            q1_error.append(
                estimate_error(BinaryHypercubeChain, d, m, sigma, beta, iter)
            )
        print(
            f"Mean Squared Error Question 1, beta={beta}, iter={iter}: {np.mean(q1_error)} +- {np.std(q1_error)}"
        )

    q1_error_sa = []
    for i in range(5):
        q1_error_sa.append(
            estimate_error_sa(BinaryHypercubeChain, d, m, sigma, betas, iter)
        )
    print(
        f"Mean Squared Error Question 1 (SA), iter={iter}: {np.mean(q1_error_sa)} +- {np.std(q1_error_sa)}"
    )
