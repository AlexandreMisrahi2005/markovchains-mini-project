import numpy as np
from tqdm import tqdm


class MetropolisHastings:
    def __init__(self, chain, iter=100):
        """
        Input args:
                basechain: a class like BinaryHypercubeChain that inherits from MarkovChain
                problem: an instance of P class
        Output args:
                pass
        """
        self.chain = chain
        self.iter = iter

    def run(self):
        samples = []
        samples.append(self.chain.current_state)

        for _ in tqdm(range(self.iter)):

            # iterate - go to next state
            self.chain.update_state()

            samples.append(self.chain.current_state)

        return np.array(samples)
