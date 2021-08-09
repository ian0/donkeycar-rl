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
        value = np.random.random()
        self.logger.record('random_value', value)
        distance = self.locals['infos'][0].get('distance')
        self.logger.log(f'distace travelled: {distance}')

        #distance = np.add(np.sq)

        return True

