import numpy as np
from abc import ABC, abstractmethod

class MarkovChain(ABC):
	"""
	Abstract class for implementing base chains
	"""
    def __init__(self, initial_state):
        self.current_state = initial_state

    @abstractmethod
    def propose(self):
        pass

    @abstractmethod
    def acceptance_probability(self, proposed_state):
        pass

    def update_state(self, proposed_state):
        if np.random.rand() < self.acceptance_probability(proposed_state):
            self.current_state = proposed_state

class BinaryHypercubeChain(MarkovChain):
	def __init__(self, d):
		self.d = d
		self.theta = np.random.randint(0, 2, size=d)

	def step(self):
        next_theta = np.copy(self.theta)
        random_idx = np.random.randint(self.d)  # Randomly select a bit to flip
        next_theta[random_idx] = 1 - next_theta[random_idx]  # Flip bit
        self.theta = next_theta

    def acceptance_probability(self, proposed_state):
        pass

class P:
	def __init__(self, X, y, beta=1., num_samples=100):
		self.X = X  # Sensing matrix
        self.y = y  # Observed measurements
        self.beta = beta  # Inverse temperature
        self.num_samples = num_samples  # Number of samples to generate

    def likelihood(self, theta):
        # Calculate the likelihood function Prob{y|theta, X}
        likelihood = theta.T @ theta # change to real likelihood
        return likelihood