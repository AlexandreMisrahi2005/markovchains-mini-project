import numpy as np

class MetropolisHastings:
	def __init__(self, basechain, problem, initial_theta=None, iter=100):
		"""
		Input args:
			basechain: a class like BinaryHypercubeChain that inherits from MarkovChain
			problem: an instance of P class
		Output args:
			pass
		"""
		self.basechain = basechain
		self.p = problem
		self.initial_theta = initial_theta if initial_theta is not None else np.random.randint(0, self.p.y.shape[0], size=self.basechain.d)
		self.iter = iter

	def run(self):
		samples = []
		current_theta = self.initial_theta

		for _ in range(self.iter):

			# iterate - go to next state
			self.basechain.step()

			# Compute new acceptance probability

			# Accept or reject the proposal

		return np.array(samples)