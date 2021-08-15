# history wrapper from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/wrappers.py

import gym
import numpy as np
from loguru import logger
from scipy.stats import norm
from simple_pid import PID
import cv2

from gym_donkeycar.envs.donkey_env import DonkeyUnitySimContoller

ACTION_STEERING = 0
ACTION_THROTTLE = 1
IMAGE_WIDTH = 160
IMAGE_HEIGHT = 120


def make_wrappers(env: gym.Env, vae) -> gym.Env:
    """Create the final set of wrappers. For consistency across scripts. DRY.

    Parameters
    ----------
    env : gym.Env
        The original environment.
    vae :
        The vae instance

    Returns
    -------
    gym.Env
        The wrapped environment.
    """
    env = VAEWrapper(env, vae)
    env = DonkeyCarActionWrapper(env, max_throttle=0.15, max_steering_angle=0.5)
    env = RenderWrapper(env)
    env = NoSteeringAtStartWrapper(env, 80)
    # env = MaxTimeStepsSafetyValve(env, 10000)
    env = HistoryWrapper(env, horizon=5)
    env = BufferHistoryWrapper(env, 10)
    return env


class VAEWrapper(gym.ObservationWrapper):
    def __init__(self, env, vae):
        gym.ObservationWrapper.__init__(self, env)
        self.vae = vae
        self.observation_space = gym.spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(1, self.vae.z_size),
            dtype=np.float32,
        )
        self.raw_observation = None
        self.log_video_size = True

    def observation(self, observation):
        self.raw_observation = observation.copy()
        # When I generate videos I set the image sensor size to be bigger.
        # But the VAE expects the same shape as it was trained with.
        if observation.shape[0] != IMAGE_HEIGHT and observation.shape[1] != IMAGE_WIDTH:
            if self.log_video_size:
                logger.debug(
                    f"The video size is not what the VAE expects. Resizing {observation.shape} to {(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}"
                )
                self.log_video_size = False
            observation = cv2.resize(
                observation, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA
            )
        return self.vae.encode(observation)


class MaxActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, max_action: float):
        super(MaxActionWrapper, self).__init__(env)
        self.max_action = max_action
        self.action_space = gym.spaces.Box(
            low=np.zeros(env.action_space.shape) - max_action,
            high=np.zeros(env.action_space.shape) + max_action,
            dtype=np.float32,
        )

    def action(self, action):
        return np.clip(action, -self.max_action, self.max_action)


class DonkeyCarActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, max_throttle: float, max_steering_angle: float):
        super(DonkeyCarActionWrapper, self).__init__(env)
        self.max_throttle = max_throttle
        self.max_steering_angle = max_steering_angle
        self.action_space = gym.spaces.Box(
            low=np.array([-self.max_steering_angle, 0]),
            high=np.array([self.max_steering_angle, self.max_throttle]),
            dtype=np.float32,
        )

    def action(self, action):
        action[ACTION_STEERING] = np.clip(
            action[ACTION_STEERING], -self.max_steering_angle, self.max_steering_angle
        )
        action[ACTION_THROTTLE] = np.clip(action[ACTION_THROTTLE], 0, self.max_throttle)
        return action


class NoSteeringAtStartWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, n_steps=80):
        super(NoSteeringAtStartWrapper, self).__init__(env)
        self.n_steps = n_steps
        self.counter = 0

    # def reset(self, **kwargs):
    #     #self.n_steps = 40
    #     self.counter = 0
    #     logger.debug("********************resetting counter*****************")
    #     pass
    #     #return self.env.reset(**kwargs)

    def action(self, action):
        # print(f'counter: {self.counter}, n_steps: {self.n_steps}')
        if self.counter < self.n_steps:
            action[ACTION_STEERING] = 0
            self.counter += 1
        elif self.counter == self.n_steps:
            logger.debug("Enabling steering")
            self.counter += 1
        return action


class SmallSteeringRewardWrapper(gym.RewardWrapper):
    def __init__(self, env: gym.Env):
        super(SmallSteeringRewardWrapper, self).__init__(env)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward, action), done, info

    def reward(self, reward, action):
        """Prefer zero angle, weighted by a gaussian normalised to 1.0."""
        angle = action[ACTION_STEERING]
        if reward > 0:
            logger.debug((reward, angle, norm.pdf(angle), reward * norm.pdf(angle)))
            return reward * norm.pdf(angle)
        else:
            return reward


class PIDActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env, p: float = 0.0, i: float = 0.1, d: float = 0.0):
        super(PIDActionWrapper, self).__init__(env)
        self.p = p
        self.i = i
        self.d = d

    def reset(self, **kwargs):
        self.pid = PID(self.p, self.i, self.d, sample_time=1, setpoint=0.0)
        self.previous_steering_angle = 0
        return self.env.reset(**kwargs)

    def action(self, action):
        next_action = self.pid(self.previous_steering_angle, dt=1)

        # Update desired steering angle, if necessary
        self.pid.setpoint = action[ACTION_STEERING]

        logger.debug(f"Requested {action[ACTION_STEERING]}, damped {next_action}")

        # Set the current steering action to that provided by the PID
        action[ACTION_STEERING] = next_action

        # Update the history for the next time
        self.previous_steering_angle = next_action
        return action


class MaxTimeStepsSafetyValve(gym.Wrapper):
    """Sometimes episodes go on for too long and get stuck. This safety valve prevents that from happening."""

    def __init__(self, env: gym.Env, max_timesteps=2000):
        """Instantiate class.

        Parameters
        ----------
        env : gym.Env
            The environment
        max_timesteps : int, optional
            The maximum number of timesteps the environment is allowed to run for, by default 2000
        """
        super(MaxTimeStepsSafetyValve, self).__init__(env)
        self.max_timesteps = max_timesteps
        self.counter = 0

    def reset(self, **kwargs):
        self.counter = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.counter += 1
        if self.counter > self.max_timesteps:
            done = True
        return observation, reward, done, info


class RenderWrapper(gym.Wrapper):
    """Donkey doesn't render properly. This fixes that."""

    def __init__(self, env: gym.Env):
        super(RenderWrapper, self).__init__(env)

    def render(self, mode="rgb_array"):
        return self.env.raw_observation


class BufferHistoryWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferHistoryWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(
            self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class HistoryWrapper(gym.Wrapper):
    """
    Stack past observations and actions to give an history to the agent.
    :param env: (gym.Env)
    :param horizon: (int) Number of steps to keep in the history.
    """

    def __init__(self, env: gym.Env, horizon: int = 5):
        assert isinstance(env.observation_space, gym.spaces.Box)

        wrapped_obs_space = env.observation_space
        wrapped_action_space = env.action_space

        # TODO: double check, it seems wrong when we have different low and highs
        low_obs = np.repeat(wrapped_obs_space.low, horizon, axis=-1)
        high_obs = np.repeat(wrapped_obs_space.high, horizon, axis=-1)

        low_action = np.repeat(wrapped_action_space.low, horizon, axis=-1)
        high_action = np.repeat(wrapped_action_space.high, horizon, axis=-1)

        low = np.concatenate((low_obs, np.expand_dims(low_action, axis=0)), axis=1)
        high = np.concatenate((high_obs, np.expand_dims(high_action, axis=0)), axis=1)

        # Overwrite the observation space
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=wrapped_obs_space.dtype)

        super(HistoryWrapper, self).__init__(env)

        self.horizon = horizon
        self.low_action, self.high_action = low_action, high_action
        self.low_obs, self.high_obs = low_obs, high_obs
        self.low, self.high = low, high
        self.obs_history = np.zeros(low_obs.shape, low_obs.dtype)
        self.action_history = np.zeros(np.expand_dims(self.low_action, axis=0).shape, low_action.dtype)
        #self.action_history = np.zeros(low_action.shape, low_action.dtype)

    def _create_obs_from_history(self):
        return np.concatenate((self.obs_history, self.action_history), axis=1)

    def reset(self):
        # Flush the history
        self.obs_history[...] = 0
        self.action_history[...] = 0
        obs = self.env.reset()
        self.obs_history[..., -obs.shape[-1]:] = obs
        return self._create_obs_from_history()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        last_ax_size = obs.shape[-1]

        self.obs_history = np.roll(self.obs_history, shift=-last_ax_size, axis=-1)
        self.obs_history[..., -obs.shape[-1]:] = obs

        self.action_history = np.roll(self.action_history, shift=-action.shape[-1], axis=-1)
        self.action_history[..., -action.shape[-1]:] = action
        return self._create_obs_from_history(), reward, done, info
