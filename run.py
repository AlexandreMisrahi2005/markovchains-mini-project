import numpy as np
import matplotlib.pyplot as plt

from MH import *
from markovchain import *

if __name__ == "__main__":
	np.random.seed(99)
	d = 10
	m = 100
	X = np.random.randn(m, d)
	true_theta = np.random.randint(0, 2, size=d)
	noise = np.random.randn(m)
	y = np.dot(X, true_theta) + noise

	mcmc = MarkovChainMonteCarlo(X, y, beta=1.0, num_samples=1000)

	initial_theta = np.random.randint(0, 2, size=d)
	samples = mcmc.metropolis_hastings(initial_theta)

	mse = 2 * np.mean(np.linalg.norm(samples.mean(axis=0) - true_theta)**2) / d
	print(f"Mean Squared Error: {mse}")