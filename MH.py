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
        samples.append(self.chain.current_state)

        for _ in tqdm(range(self.iter)):

            # iterate - go to next state
            self.chain.update_state()

            samples.append(self.chain.current_state)

        return np.array(samples)
