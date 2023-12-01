import numpy as np
from tqdm import tqdm

from markovchain import *


class MetropolisHastings:
    def __init__(self, chain, d, initial_state, X, y, beta=1.0, iter=100):
        """
        Input args:
                chain: a chain class like BinaryHypercubeChain that inherits from MarkovChain
                d: int, dimensionality of the problem
                initial_state: numpy array of size (d,)
                X: numpy array of size (m,d)
                y: numpy array of size (m,)
                beta: float > 0, represents inverse temperature
                iter: int, number of state updates in the base chain
        Returns:
                samples: numpy array of size (iter,)
        """

        acceptance_calc = AcceptanceCalculator(X, y, beta)
        self.chain = chain(acceptance_calc, d, initial_state)
        self.iter = iter

    def run(self):
        """
        Runs the MH algorithm for self.iter number of steps
        """
        samples = []
        samples.append(self.chain.current_state.copy())

        for _ in tqdm(range(self.iter)):
            # iterate - go to next state
            self.chain.update_state()

            samples.append(self.chain.current_state.copy())

        return samples


class SimulatedAnnealing:
    def __init__(
        self, chain, d, initial_state, X, y, betas=np.logspace(0, 10, 10), iter=100
    ):
        """
        Input args:
                chain: a chain class like BinaryHypercubeChain that inherits from MarkovChain
                d: int, dimensionality of the problem
                initial_state: numpy array of size (d,)
                X: numpy array of size (m,d)
                y: numpy array of size (m,)
                betas: list[float], represents inverse temperatures for the simulated annealing
                iter: int, number of state updates in the base chain (one beta will have iter//len(betas) iterations)
        Returns:
                samples: numpy array of size (iter,)
        """
        self.betas = betas
        self.iter = iter
        self.chain_class = chain
        self.d = d
        self.initial_state = initial_state
        self.X = X
        self.y = y

    def run(self):
        """
        Runs the simulated annealing algorithm for self.iter number of steps
        """
        samples = []

        starting_state = self.initial_state
        iter = self.iter // len(self.betas)

        samples.append(starting_state)

        for beta in tqdm(self.betas):
            acceptance_calc = AcceptanceCalculator(self.X, self.y, beta)
            self.chain = self.chain_class(acceptance_calc, self.d, starting_state)
            for _ in range(iter):
                # iterate - go to next state
                self.chain.update_state()

                samples.append(self.chain.current_state.copy())
            starting_state = self.chain.current_state

        return samples
