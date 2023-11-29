import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm


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
    def accept_proposed_state(self):
        pass

    def update_state(self):
        self.propose()
        if np.random.rand() < self.acceptance_probability():
            self.accept_proposed_state()


class AcceptanceCalculator:
    def __init__(self, X, y, beta=1.0):
        self.X = X  # Sensing matrix
        self.y = y  # Observed measurements
        self.beta = beta  # Inverse temperature
        self.noise = None
        self.proposed_noise = None

    def acceptance(self, theta, flip_idx):
        if self.noise is None:
            self.noise = self.X @ theta - self.y

        theta_idx = theta[flip_idx]
        coef = 1 - 2 * theta_idx

        X_idx = self.X[:, flip_idx]

        self.proposed_noise = self.noise + coef * X_idx

        power = -self.beta * coef * np.dot(2 * self.noise + coef * X_idx, X_idx)
        if power >= 0:
            return 1
        return np.exp(power)

    def swap_acceptance(self, theta, flip_one_idx, flip_zero_idx):
        if self.noise is None:
            self.noise = self.X @ theta - self.y

        X_one_idx = self.X[:, flip_one_idx]
        X_zero_idx = self.X[:, flip_zero_idx]

        X_idx = X_zero_idx - X_one_idx

        self.proposed_noise = self.noise + X_idx

        power = -self.beta * np.dot(2 * self.noise + X_idx, X_idx)
        if power >= 0:
            return 1
        return np.exp(power)
    
    def update_noise(self):
        self.noise = self.proposed_noise


class SignLikelihoodAcceptanceCalculator:
    def __init__(self, X, y, beta=1.0, sigma=1.0):
        self.X = X*y.reshape(-1, 1)  # Sensing matrix
        self.beta = beta  # Inverse temperature
        self.noise = None
        self.proposed_noise = None
        self.cdf = norm(scale=sigma).cdf

    def swap_acceptance(self, theta, flip_one_idx, flip_zero_idx):
        if self.noise is None:
            self.noise = self.X @ theta 

        X_one_idx = self.X[:, flip_one_idx]
        X_zero_idx = self.X[:, flip_zero_idx]

        X_idx = X_zero_idx - X_one_idx

        self.proposed_noise = self.noise + X_idx

        prob = np.prod(np.array(list(map(self.cdf, self.proposed_noise))) / np.array(list(map(self.cdf, self.noise)))) ** self.beta
        if prob >= 1:
            return 1
        return prob ** self.beta
    
    def update_noise(self):
        self.noise = self.proposed_noise


class BinaryHypercubeChain(MarkovChain):
    def __init__(self, acceptance_calc, d, initial_theta=None):
        self.acceptance_calc = acceptance_calc
        self.d = d

        self.current_state = (
            initial_theta
            if initial_theta is not None
            else np.random.randint(0, 2, size=d)
        )

        self.flip_idx = None

    def propose(self):
        self.flip_idx = np.random.randint(self.d)

    def acceptance_probability(self):
        return self.acceptance_calc.acceptance(self.current_state, self.flip_idx)

    def accept_proposed_state(self):
        self.current_state[self.flip_idx] = 1 - self.current_state[self.flip_idx]

        self.acceptance_calc.update_noise()


class SwapChain(MarkovChain):
    def __init__(self, acceptance_calc, d, initial_theta=None):
        self.acceptance_calc = acceptance_calc
        self.d = d

        self.current_state = (
            initial_theta
            if initial_theta is not None
            else np.random.randint(0, 2, size=d)
        )

        self.flip_one_idx, self.flip_zero_idx = None, None

    def propose(self):
        ones = np.argwhere(self.current_state == 1).reshape(-1)
        zeros = np.argwhere(self.current_state == 0).reshape(-1)

        self.flip_one_idx = np.random.choice(ones)
        self.flip_zero_idx = np.random.choice(zeros)

    def acceptance_probability(self):
        return self.acceptance_calc.swap_acceptance(
            self.current_state, self.flip_one_idx, self.flip_zero_idx
        )

    def accept_proposed_state(self):
        self.current_state[self.flip_zero_idx] = 1
        self.current_state[self.flip_one_idx] = 0

        self.acceptance_calc.update_noise()
