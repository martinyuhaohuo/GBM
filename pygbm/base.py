
class GBM_setup:
    """
    Base configuration class for a Geometric Brownian Motion (GBM) process.

    Attributes:
        y0 (float): initial value of the process at time t=0.
        mu (float): drift parameter of the GBM.
        sigma (float): diffusion (volatility) parameter of the GBM.
    """
    
    def __init__(self, y0, mu, sigma):
        """
        Initializes the simulator setup.

        Parameters:
            y0 (float): initial value of the process at time t=0.
            mu (float): drift parameter of the GBM.
            sigma (float): diffusion (volatility) parameter of the GBM.
        """
        self.y0 = y0 #initial value
        self.mu = mu #drift
        self.sigma = sigma #diffusion coefficient