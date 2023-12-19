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

    question = 0
    # question = 1.6
    # question = 2.3
    # question = 3.2

    d = 2000
    s = d // 100
    # m = 200
    sigma = 1

    beta = 0.1

    ###############################
    ### TEST                    ###
    ###############################

    if question == 0:
        m = 1000
        e = estimate_error(BinaryHypercubeChain, d, m, sigma, beta)
        print(e)

    ###############################
    ### Q. 1.1.6                ###
    ###############################

    if question == 1.6:

        # create a dense enough array of values of m
        m_array = sorted(list(set([int(d*(1-0.025*r)) for r in range(1, 30, 2)] + [int(d*(1+0.125*r)) for r in range(0, 20, 2)])))
        errors = []
        runs = 5
        for m in m_array:
            print("m =", m)
            errors_m = []
            for i in range(runs):
                error = 2 / d * estimate_error(BinaryHypercubeChain, d, m, sigma, beta)
                errors_m.append(error)
            print("m/d =", m/d, " error =", np.mean(errors_m))
            print("")
            errors.append(np.mean(errors_m))
        
        plt.plot(m_array, errors)
        plt.title(f"Expected error over {runs}-fold {100*d} iterations \nof MH for each $m$, $d$ = {d}")
        plt.xlabel("$m$")
        plt.ylabel("Error")
        plt.savefig(f"1-1-6-d={d}.png")
        plt.show()

    ###############################
    ### Q. 1.2.3                ###
    ###############################

    if question == 2.3:

        m_array = np.array([20, 50, 100, 130, 150, 155, 175, 200])
        errors = []
        runs = 5
        for m in m_array:
            print("m =", m)
            errors_m = []
            for i in range(runs):
                error = estimate_error(SwapChain, d, m, sigma, beta, s) / (2 * s)
                errors_m.append(error)
            print("m/d =", m/d, " error =", np.mean(errors_m))
            print("")
            errors.append(np.mean(errors_m))
        
        plt.plot(m_array, errors)
        plt.title(f"SwapChain expected error over {runs}-fold {100*d} iterations \nof MH for each $m$, $d$ = {d}, $s$ = {s}")
        plt.xlabel("$m$")
        plt.ylabel("Error")
        plt.savefig(f"1-2-3-d={d}.png")
        plt.show()

    ###############################
    ### Q. 1.3.2                ###
    ###############################

    if question == 3.2:

        m_array = np.array([20, 50, 100])
        errors = []
        runs = 5
        for m in m_array:
            print("m =", m)
            errors_m = []
            for i in range(runs):
                error = estimate_error(SwapChain, d, m, sigma, beta, s, True) / (2 * s)
                errors_m.append(error)
            print("m/d =", m/d, " error =", np.mean(errors_m))
            print("")
            errors.append(np.mean(errors_m))
        
        plt.plot(m_array, errors)
        plt.title(f"SwapChain expected error over {runs}-fold {100*d} iterations \nof MH for each $m$, $d$ = {d}, $s$ = {s}")
        plt.xlabel("$m$")
        plt.ylabel("Error")
        plt.savefig(f"1-2-3-d={d}.png")
        plt.show()
