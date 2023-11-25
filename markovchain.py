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
    def acceptance_probability(self):
        pass

    @abstractmethod
    def proposed_state(self):
        pass

    def update_state(self):
        self.propose()
        if np.random.rand() < self.acceptance_probability():
            self.current_state = self.proposed_state()


class AcceptanceCalculator:
    def __init__(self, X, y, beta=1.0):
        self.X = X  # Sensing matrix
        self.y = y  # Observed measurements
        self.beta = beta  # Inverse temperature

    def acceptance(self, theta, flip_idx):
        noise = self.X @ theta - self.y

        theta_idx = theta[flip_idx]
        coef = 1 - 2 * theta_idx

        X_idx = self.X[:, flip_idx]

        power = -self.beta * coef * np.dot(2 * noise + coef * X_idx, X_idx)
        if power >= 0:
            return 1
        return np.exp(power)


class BinaryHypercubeChain(MarkovChain):
    def __init__(self, acceptance_calc, d, initial_theta=None):
        self.acceptance_calc = acceptance_calc
        self.d = d

        self.current_state = (
            initial_theta
            if initial_theta is not None
            else np.random.randint(0, 2, size=d)
        )

        self.flip_idx = 0

    def propose(self):
        self.flip_idx = np.random.randint(self.d)

    def acceptance_probability(self):
        return self.acceptance_calc.acceptance(self.current_state, self.flip_idx)

    def proposed_state(self):
        theta = self.current_state
        theta[self.flip_idx] = 1 - self.current_state[self.flip_idx]
        return theta
