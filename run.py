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
    iter = 10 * d

    X, true_theta, y = generate_data(d, m, sigma, s)

    initial_theta = generate_theta(d, s)

    mh = MetropolisHastings(chain_type, d, initial_theta, X, y, beta, iter)
    mh.run()  # compute samples

    estimate_theta = mh.chain.current_state

    return np.linalg.norm(estimate_theta - true_theta) ** 2 * 2 / d


def estimate_error_sa(chain_type, d, m, sigma, betas, s=None):
    iter = 10 * d

    X, true_theta, y = generate_data(d, m, sigma, s)

    initial_theta = generate_theta(d, s)

    samh = SimulatedAnnealing(chain_type, d, initial_theta, X, y, betas, iter)
    samh.run()  # compute samples

    estimate_theta = samh.chain.current_state

    return np.linalg.norm(estimate_theta - true_theta) ** 2 * 2 / d

if __name__ == "__main__":
    d = 2000
    s = d // 100
    m = 2000
    sigma = 1

    beta = 0.1 # Without SA
    betas = np.logspace(-1, 40, num=40) # SA


    q1_error = []
    for i in range(3):
        q1_error.append(estimate_error(BinaryHypercubeChain, d, m, sigma, beta))


    # naive_q2_error = []
    # for i in range(3):
    #     naive_q2_error.append(estimate_error(NaiveSwapChain, d, m, sigma, beta, s))

    # q2_error = []
    # for i in range(3):
    #     q2_error.append(estimate_error(SwapChain, d, m, sigma, beta, s))


    q1_error_sa = []
    for i in range(3):
        q1_error_sa.append(estimate_error_sa(BinaryHypercubeChain, d, m, sigma, betas))


    # naive_q2_error_sa = []
    # for i in range(3):
    #     naive_q2_error_sa.append(estimate_error_sa(NaiveSwapChain, d, m, sigma, betas, s))

    # q2_error_sa = []
    # for i in range(3):
    #     q2_error_sa.append(estimate_error_sa(SwapChain, d, m, sigma, betas, s))


    print(f"Mean Squared Error Question 1: {np.mean(q1_error)}")
    print(f"Mean Squared Error Question 1 (SA): {np.mean(q1_error_sa)}")

    # print(f"Mean Squared Error Question 2 (naive): {np.mean(naive_q2_error)}")
    # print(f"Mean Squared Error Question 2 (naive, SA): {np.mean(naive_q2_error_sa)}")

    # print(f"Mean Squared Error Question 2: {np.mean(q2_error)}")
    # print(f"Mean Squared Error Question 2 (SA): {np.mean(q2_error_sa)}")
