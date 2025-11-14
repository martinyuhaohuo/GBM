import numpy as np
import matplotlib.pyplot as plt
from .base import GBM_setup

class GBM_simulator(GBM_setup):
    def __init__(self, y0, mu, sigma):
        super().__init__(y0, mu, sigma)
    
    def simulate_path(self, T, N, seed = 18769):
        t_values = np.linspace(0, T, N+1)
        y_values = [self.y0]
        B_values = [0]
        dt = T/N
        rng = np.random.default_rng(seed)
        for t in t_values[1:]:
            B_values.append(B_values[-1] + rng.normal(0,np.sqrt(dt)))
            B_t = B_values[-1]
            y_values.append(
                y_values[0]*np.exp((self.mu - 0.5*(self.sigma**2))*t + self.sigma*B_t)
                )
        return t_values, y_values
    
    def plot_path(self, t_values, y_values):
        plt.plot(t_values , y_values , label ="GBM Path ")
        plt.xlabel("Time")
        plt.ylabel("Y(t)")
        plt.title("Simulated Geometric Brownian Motion Path")
        plt.legend()
        plt.show()
