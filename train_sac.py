import argparse

import gym
#import tensorflow as tf
from loguru import logger
from stable_baselines3 import SAC
#from models.custom_sac import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.sac import MlpPolicy
import torch
import yaml

import gym_donkeycar
from environment.utility import seed
from environment.wrappers import make_wrappers
from environment.command import common_args, parse_args
import logging
import time
from environment.controller import AEController

from callbacks import TensorboardCallback
from environment.custom_reward import reward2, reward3

from stable_baselines3.common.evaluation import evaluate_policy

def load_ae_controller(path=None):
    ae_controller = AEController(path)
    return ae_controller


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
    #env.set_reward_fn(reward3)
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
        id = int(time.time())
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/track/",
                                                 name_prefix="donkey_model")
        callback = CallbackList([checkpoint_callback, test_callback])

        seed(42, env)

        policy = dict(activation_fn=torch.nn.ReLU, net_arch=[64, 64], use_sde=True, log_std_init=-2)

        logger.info('create model and start learning')

        model = SAC(MlpPolicy,
                    policy_kwargs=policy,
                    env=env,
                    verbose=1,
                    batch_size=64,
                    buffer_size=30000,
                    learning_starts=0,
                    gradient_steps=64,
                    train_freq=(1, "episode"),
                    ent_coef='auto',
                    learning_rate=3e-4,
                    tensorboard_log=str(args["tensorboard_dir"]),
                    gamma=0.99,
                    tau=0.02,
                    use_sde_at_warmup=True,
                    use_sde=True,
                    sde_sample_freq=64,
                    )

        model.learn(total_timesteps=int(50000), callback=callback)

        logger.info('save the model')
        # save the model
        model.save("sac_donkeycar")

        logger.info('save the replay buffer')
        # now save the replay buffer too
        model.save_replay_buffer("sac_donkeycar_replay_buffer")


    except KeyboardInterrupt as e:
        logging.info('Finished early')
        pass
    finally:
        # logger.info(f'Trained for {env.get_total_steps()}')
        logger.info(f'Saving model to {args["model_path"]}, don\'t quit!')
        model.save(args["model_path"])
        env.close()
        logging.info('Finished')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the SAC algorithm on the DonkeyCar environment with a VAE."
    )
    parser = common_args(parser)
    main(parse_args(parser))
