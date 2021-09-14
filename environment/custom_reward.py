
import math
import logging
from configs.config import REWARD_CRASH, CRASH_REWARD_WEIGHT, THROTTLE_REWARD_WEIGHT, MAX_THROTTLE, MIN_THROTTLE


logger = logging.getLogger(__name__)

def reward2(self, done):

    #logger.info('In reward2')
    if done:
        return REWARD_CRASH + CRASH_REWARD_WEIGHT * self.throttle_average
    throttle_reward = THROTTLE_REWARD_WEIGHT * (self.last_throttle / MAX_THROTTLE)
    return 1 + ((1.0 - (math.fabs(self.cte) / self.max_cte)) * throttle_reward)

def reward3(self, done):
    if done:
        # penalize the agent for getting off the road fast
        norm_throttle = (self.last_throttle - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE)
        return REWARD_CRASH - CRASH_REWARD_WEIGHT * norm_throttle
    # 1 per timesteps + throttle
    throttle_reward = THROTTLE_REWARD_WEIGHT * (self.last_throttle / MAX_THROTTLE)
    return 1 + ((1.0 - (math.fabs(self.cte) / self.max_cte)) * throttle_reward)

def reward4(self, done):
    if done:
        return REWARD_CRASH + CRASH_REWARD_WEIGHT * self.throttle_average
    throttle_reward = THROTTLE_REWARD_WEIGHT * (self.throttle_average / MAX_THROTTLE)
    return 1 + ((1.0 - (math.fabs(self.cte) / self.max_cte)) * throttle_reward)
