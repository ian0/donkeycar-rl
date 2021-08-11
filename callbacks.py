import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        distance = self.locals['infos'][0].get('distance')
        speed = self.locals['infos'][0].get('speed')
        self.logger.record('distance_travelled', distance)
        self.logger.record('speed', speed)
        return True
