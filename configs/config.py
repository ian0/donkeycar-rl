#
#
# REWARD_CRASH, CRASH_REWARD_WEIGHT, THROTTLE_REWARD_WEIGHT, MAX_THROTTLE, MIN_THROTTLE
#

#
# # Reward parameters
THROTTLE_REWARD_WEIGHT = 0.2
# JERK_REWARD_WEIGHT = 0.0
#
# # very smooth control: 10% -> 0.2 diff in steering allowed (requires more training)
# # smooth control: 15% -> 0.3 diff in steering allowed
# MAX_STEERING_DIFF = 0.15
# Negative reward for getting off the road
REWARD_CRASH = -10
# Penalize the agent even more when being fast
CRASH_REWARD_WEIGHT = 5
#
# Symmetric command
MAX_STEERING = 1
MIN_STEERING = - MAX_STEERING

# Simulation config
MIN_THROTTLE = 0.4
# max_throttle: 0.6 for level 0 and 0.5 for level 1
MAX_THROTTLE = 0.6
# Number of past commands to concatenate with the input
N_COMMAND_HISTORY = 20
# Max cross track error (used in normal mode to reset the car)
MAX_CTE_ERROR = 2.0
# Level to use for training
LEVEL = 0

STEERING_THRESHOLD = 0.15
