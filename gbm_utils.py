import numpy as np

class GBMSimulation:
	def __init__(self, initial_price, mu, sigma, days=252, simulations=10):
		self.SO = initial_price
		self.mu = mu
		self.sigma = sigma
		self.days = days
		self.simualtions = simulations
		self.simulated_paths = None

	def adjust_for_viewpoints(self, viewpoint):
		"""
		the hybrid mathemathical model mixed with machine learning is met with
		additional computation (my estimate) the user has to input either 
		bullish, bearish, volatile, stable only
		"""
		if viewpoint == "bullish":
			self.mu *= 1.2
		elif viewpoint == "bearish":
			self.mu *= 0.8
		elif viewpoint == "stable":
			self.sigma *= 0.8
		elif viewpoint == "volatile":
			self.sigma *= 1.5 

	def simulate(self):
		"""
		Simulates the Geometric Brownian Motion of the asset in the linear space specified by the user
		"""

		dt = 1 / self.days
		paths = np.zeros((self.days, self.simulations))
		paths[0] = self.SO

		for t in range(1, self.days+1):
			Z = np.random.standard_normal(self.simulations)
			paths[t] = paths[t - 1] * np.exp((self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z)

		self.simulated_paths = paths
		return paths


