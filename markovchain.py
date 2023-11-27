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
        new_state, acceptance_prob = self.propose()
        if np.random.rand() < acceptance_prob:
            self.current_state = new_state


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

    def swap_acceptance(self, theta, flip_one_idx, flip_zero_idx):
        noise = self.X @ theta - self.y

        X_one_idx = self.X[:, flip_one_idx]
        X_zero_idx = self.X[:, flip_zero_idx]

        X_idx = X_zero_idx - X_one_idx

        power = -self.beta * np.dot(2 * noise + X_idx, X_idx)
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

    def propose(self):
        flip_idx = np.random.randint(self.d)
        return self.proposed_state(flip_idx), self.acceptance_probability(flip_idx)

    def acceptance_probability(self, flip_idx):
        return self.acceptance_calc.acceptance(self.current_state, flip_idx)

    def proposed_state(self, flip_idx):
        theta = self.current_state.copy()
        theta[flip_idx] = 1 - self.current_state[flip_idx]
        return theta


class SwapChain(MarkovChain):
    def __init__(self, acceptance_calc, d, initial_theta=None):
        self.acceptance_calc = acceptance_calc
        self.d = d

        self.current_state = (
            initial_theta
            if initial_theta is not None
            else np.random.randint(0, 2, size=d)
        )

    def propose(self):
        ones = np.argwhere(self.current_state == 1).reshape(-1)
        zeros = np.argwhere(self.current_state == 0).reshape(-1)

        flip_one_idx = np.random.choice(ones)
        flip_zero_idx = np.random.choice(zeros)

        return self.proposed_state(flip_one_idx, flip_zero_idx), self.acceptance_probability(flip_one_idx, flip_zero_idx)

    def acceptance_probability(self, flip_one_idx, flip_zero_idx):
        return self.acceptance_calc.swap_acceptance(
            self.current_state, flip_one_idx, flip_zero_idx
        )

    def proposed_state(self, flip_one_idx, flip_zero_idx):
        theta = self.current_state.copy()
        theta[flip_zero_idx] = 1
        theta[flip_one_idx] = 0
        return theta
