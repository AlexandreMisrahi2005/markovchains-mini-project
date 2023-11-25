import numpy as np
from tqdm import tqdm

from markovchain import *


class MetropolisHastings:
    def __init__(self, chain, d, initial_state, X, y, beta=1.0, iter=100):
        """
        Input args:
                basechain: a class like BinaryHypercubeChain that inherits from MarkovChain
                problem: an instance of P class
        Output args:
                pass
        """

        acceptance_calc = AcceptanceCalculator(X, y, beta)
        self.chain = chain(acceptance_calc, d, initial_state)
        self.iter = iter

    def run(self):
        samples = []
        samples.append(self.chain.current_state)

        for _ in tqdm(range(self.iter)):

            # iterate - go to next state
            self.chain.update_state()

            samples.append(self.chain.current_state)

        return np.array(samples)
