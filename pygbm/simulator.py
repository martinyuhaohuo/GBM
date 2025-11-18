import numpy as np
import matplotlib.pyplot as plt
from .base import GBM_setup

class GBM_simulator(GBM_setup):
    """
    Simulator class for generating sample paths of a Geometric Brownian Motion.
    Inherits from GBM_setup and provides several numerical methods for GBM simulation.

    Attributes:
        y0 (float): initial value of the process at time t=0.
        mu (float): drift parameter of the GBM.
        sigma (float): diffusion (volatility) parameter of the GBM.
    """

    def __init__(self, y0, mu, sigma):
        """
        Initializes the simulator with inheritance from the GBM_setup.

        Parameters:
            y0 (float): initial value of the process at time t=0.
            mu (float): drift parameter of the GBM.
            sigma (float): diffusion (volatility) parameter of the GBM.
        """
        super().__init__(y0, mu, sigma)
    
    def exact_method(self, t_values, B_values):
        """
        Simulate GBM using the closed-form analytical solution.

        Parameters:
            t_values (ndarray): array of time points starting at 0.
            B_values (ndarray): Brownian increments accumulated as B(t).

        Returns:
            tuple: A tuple containing:
                t_values (ndarray): time grid for the simulation.
                y_values (ndarray): simulated GBM values at each time point.
        """
        y_values = self.y0*np.exp((self.mu - 0.5*(self.sigma**2))*t_values[1:] + self.sigma*B_values)
        y_values = np.insert(y_values, 0, self.y0, axis=0)
        return t_values, y_values
    
    def euler_method(self, t_values, B_values):
        """
        Simulate GBM using the Euler Maruyama approximation.

        Parameters:
            t_values (ndarray): array of time points starting at 0.
            B_values (ndarray): Brownian increments accumulated as B(t).

        Returns:
            tuple: A tuple containing:
                t_values (ndarray): time grid for the simulation.
                y_values (ndarray): simulated GBM values at each time point.
        """
        y_values = [self.y0]
        dt = t_values[-1]/(len(t_values)-1)
        dB_values =np.insert(np.diff(B_values), 0, [B_values[0]], axis=0)
        for dB in dB_values:
            y_pre = y_values[-1]
            y_values.append(y_pre + y_pre*self.mu*dt + self.sigma*y_pre*dB)
        y_values = np.array(y_values)
        return t_values, y_values
    
    def milstein_method(self, t_values, B_values):
        """
        Simulate GBM using the Milstein method.

        Parameters:
            t_values (ndarray): array of time points starting at 0.
            B_values (ndarray): Brownian increments accumulated as B(t).

        Returns:
            tuple: A tuple containing:
                t_values (ndarray): time grid for the simulation.
                y_values (ndarray): simulated GBM values at each time point.
        """
        y_values = [self.y0]
        dt = t_values[-1]/(len(t_values)-1)
        dB_values =np.insert(np.diff(B_values), 0, [B_values[0]], axis=0)
        for dB in dB_values:
            y_pre = y_values[-1]
            y_values.append(
                y_pre + y_pre*self.mu*dt + self.sigma*y_pre*dB 
                + 0.5*(self.sigma**2)*y_pre*(dB**2-dt)
                )
        y_values = np.array(y_values)
        return t_values, y_values

    def simulate_path(self, T, N, method = "exact_method", seed = None):
        """
        Generate a single GBM simulation path using a selected numerical method.

        Parameters:
            T (float): Total time horizon.
            N (int): Number of discretization steps.
            method (str, optional): name of the simulation method. 
                Options: `"exact_method"`, `"euler_method"`, `"milstein_method"`.
            seed (int, optional): random seed for reproducibility.

        Returns:
            tuple: A tuple containing:
                t_values (ndarray): time grid for the simulation.
                y_values (ndarray): simulated GBM values at each time point.
        """
        t_values = np.linspace(0, T, N+1)
        dt = T/N
        rng = np.random.default_rng(seed)
        dB =  rng.normal(0,np.sqrt(dt),size=N)
        B_values = np.cumsum(dB)
        simulation = getattr(self, method)
        t_values, y_values = simulation(t_values, B_values)
        return t_values, y_values
    
    def plot_path(self, t_values, y_values):
        """
        Plot a simulated GBM path. To use, add plt.show() after implemnting this method.

        Parameters:
            t_values (ndarray): time grid for the simulation.
            y_values (ndarray): simulated GBM values at each time point.

        Returns:
            None
        """
        plt.plot(t_values , y_values , label ="GBM Path ")
        plt.xlabel("Time")
        plt.ylabel("Y(t)")
        plt.title("Simulated Geometric Brownian Motion Path")
        plt.legend()
        # plt.show()
    
    def simulate_compare(self, T, N, seed, methods = ["exact_method", "euler_method", "milstein_method"]):
        """
        Compare multiple GBM simulation methods on the same Brownian sample path.

        Parameters:
            T (float): total time horizon.
            N (int): number of discretization steps.
            seed (int): random seed for reproducibility.
            methods (list of str, optional): list of simulation methods to compare.
        
        Returns:
            None
        """
        t_values = np.linspace(0, int(T), int(N+1))
        dt = T/N
        rng = np.random.default_rng(seed)
        dB =  rng.normal(0,np.sqrt(dt),size=N)
        B_values = np.cumsum(dB)
        for method in methods:
            simulation = getattr(self, method)
            t_values, y_values = simulation(t_values, B_values)
            plt.plot(t_values , y_values , label = method)
        plt.xlabel("Time")
        plt.ylabel("Y(t)")
        plt.title("Simulated Geometric Brownian Motion Path")
        plt.legend()
        # plt.show()
