from loguru import logger
import numpy as np
import random
# import tensorflow as tf
from environment.controller import AEController

def seed(value, env):
    logger.debug(f"Setting random seeds with value {value}.")
    np.random.seed(value)
    env.seed(value)
    env.action_space.seed(value)
    random.seed(value)
    #tf.random.set_random_seed(value)


def load_ae_controller(path=None):
    ae_controller = AEController()
    return ae_controller

