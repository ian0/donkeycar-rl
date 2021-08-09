import argparse
import time
from pathlib import Path

import gym
import numpy as np
#import tensorflow as tf
from loguru import logger
from stable_baselines3 import SAC

import gym_donkeycar
from environment.command import common_args, parse_args
from environment.plotting import VAEVideo
from environment.utility import load_ae_controller, seed
from environment.wrappers import make_wrappers
from callbacks import TensorboardCallback


#tf.logging.set_verbosity(tf.logging.ERROR)


def main(args: dict):
    vae = load_ae_controller(args["ae_path"])

    env = gym.make(args["environment_id"])
    mp4_path = args["monitoring_dir"] / time.strftime("VAE_video_%H%M%S.mp4")
    logger.debug("Saving video to {mp4_path}")
    #with VAEVideo(mp4_path, args["video_width"], args["video_height"]) as video:
    try:
        env = make_wrappers(env, vae)

        model = SAC.load(args["model_path"])
        test_callback = TensorboardCallback()

        seed(42, env)
        obs = env.reset()
        for _ in range(args["max_time_steps"]):
            #video.write_frame(env.raw_observation, vae.decode(obs)[0])
            action = model.predict(obs, deterministic=True)[0]
            obs, _, done, _ = env.step(action)
            print(f'done: {done}')
            if done:
                break
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video of a policy")
    parser = common_args(parser)
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=5000,
        help="Maximum number of timesteps to run simulation for.",
    )
    parser.add_argument(
        "-width", "--video-width", help="Width of final video", type=int, default=320,
    )
    parser.add_argument(
        "-height",
        "--video-height",
        help="Height of final video",
        type=int,
        default=240,
    )
    main(parse_args(parser))
