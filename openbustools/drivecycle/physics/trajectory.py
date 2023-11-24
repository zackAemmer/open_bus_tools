import numpy as np
import pandas as pd


class RandomVehTrajectory():
    def __init__(self) -> None:
        self.traj_len = np.random.randint(10,50)
        # Avg values between current and previous points
        self.vel = np.random.randint(1, 35, size=self.traj_len)
        self.dist = np.random.randint(1, 4200, size=self.traj_len)
        # Model will give time; from which calculate speed w/dist. Here opposite.
        self.tim = self.dist[1:] / self.vel[1:]
        # Measured at each point
        self.acc = (self.vel[1:] - self.vel[:-1]) / (self.tim)
        self.elev = np.random.randint(-200, 200, size=self.traj_len)
        self.slope = (self.elev[1:] - self.elev[:-1]) / self.dist[1:]
        self.theta = np.arctan(self.slope)
        # Calculated new avg values; lose first point
        self.vel = self.vel[1:]
        self.dist = self.dist[1:]
        self.elev = self.elev[1:]
    def to_df(self):
        df = pd.DataFrame({
            'Speed': self.vel,
            'Acceleration': self.acc,
            'Distance': self.dist,
            'Elevation': self.elev,
            'Time': self.tim,
            'Slope': self.slope,
            'Theta': self.theta})
        return df