import argparse

import gym
from loguru import logger
from stable_baselines3 import SAC
# from models.custom_sac import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.sac import MlpPolicy
import torch

import gym_donkeycar
from environment.utility import seed, load_ae_controller
from environment.wrappers import make_wrappers
from environment.command import common_args, parse_args
import logging
import time

from callbacks import TensorboardCallback

from stable_baselines3.common.evaluation import evaluate_policy


def main(args: dict):
    logging.basicConfig(level=logging.INFO)
    logging.info('Started')

    vae = load_ae_controller(args["ae_path"])

    train_conf = {"exe_path": "/home/matthewi/project/DonkeySimLinux/donkey_sim.x86_64",
                  "host": "127.0.0.1",
                  "port": 9091,
                  "car_name": "training",
                  "max_cte": 4.0
                  }

    env = gym.make(args["environment_id"], conf=train_conf)
    try:
        env = make_wrappers(env, vae)

        env.metadata["video.frames_per_second"] = 10
        env = gym.wrappers.monitor.Monitor(
            env,
            directory=args["monitoring_dir"],
            force=True,
            video_callable=lambda episode: episode % 5,  # Dump every 5 episodes
        )

        test_callback = TensorboardCallback()
        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/speed-reward",
                                                 name_prefix="donkey_model")
        callback = CallbackList([checkpoint_callback, test_callback])

        env = DummyVecEnv([lambda: env])


        logger.info('load model and start learning')
        loaded_model = SAC.load("logs/reward-2/27000.zip")
        #
        # logger.info('load replay buffer into loaded model')
        # loaded_model.load_replay_buffer("sac_donkeycar_replay_buffer")

        loaded_model.env = env


        loaded_model.learn(total_timesteps=int(10000), eval_freq=50, n_eval_episodes=5,
                            eval_log_path="./logs/speed-reward", callback=callback)



        #
        # # Save the policy independently from the model
        # # Note: if you don't save the complete model with `model.save()`
        # # you cannot continue training afterward
        # logger.info('save the policy')
        # policy = loaded_model.policy
        # policy.save("sac_policy_donkeycar")
        #
        # logger.info('Retrieve the environment')
        # env = loaded_model.get_env()
        #
        # logger.info('Evaluate the policy')
        # mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)
        #
        # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        #
        # # Load the policy independently from the model
        # saved_policy = MlpPolicy.load("sac_policy_donkeycar")
        #
        # # Evaluate the loaded policy
        # mean_reward, std_reward = evaluate_policy(saved_policy, env, n_eval_episodes=10, deterministic=True)
        #
        # print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    except ValueError as e:
        logging.info('Finished early')
        pass
    finally:
        # logger.info(f'Trained for {env.get_total_steps()}')
        # logger.info(f'Saving model to {args["model_path"]}, don\'t quit!')
        # loaded_model.save(args["model_path"])
        env.close()
        logging.info('Finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the SAC algorithm on the DonkeyCar environment with a VAE."
    )
    parser = common_args(parser)
    main(parse_args(parser))
